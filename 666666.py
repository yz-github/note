import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
from torch import distributions as dis
import torch.nn.functional as F

EPS = 1e-6          # Avoid NaN (prevents division by zero or log of zero)
SIG_MIN = 1e-3


class VRM(nn.Module):
    def __init__(self,
                 input_size,
                 action_size,
                 rnn_type='mtlstm',
                 d_layers=[256],
                 z_layers=[64],
                 taus=[1.0, ],
                 decode_layers=[128, 128],
                 x_phi_layers=[128, ],
                 posterior_layers=[128, ],
                 prior_layers=[128, ],
                 lr_st=8e-4,
                 optimizer='adam',
                 feedforward_actfun_rnn=nn.Tanh,
                 sig_scale='auto',
                 device="cpu"):

        super(VRM, self).__init__()

        if len(d_layers) != len(taus):
            raise ValueError("Length of hidden layer size and timescales should be the same.")

        self.input_size = input_size
        self.action_size = action_size
        self.rnn_type = rnn_type
        self.d_layers = d_layers
        self.z_layers = z_layers
        self.taus = taus        
        self.decode_layers = decode_layers
        self.x_phi_layers = x_phi_layers
        self.posterior_layers = posterior_layers
        self.prior_layers = prior_layers
        self.sig_scale = sig_scale
        self.device = device

        self.n_levels = len(d_layers)
        self.action_feedback = False
        self.batch = True
                
        # feature-extracting transformations
        self.x2phi = nn.ModuleList()
        last_layer_size = self.input_size
        for layer_size in self.x_phi_layers:
            self.x2phi.append(nn.Linear(last_layer_size, layer_size, bias=True))
            self.x2phi.append(feedforward_actfun_rnn())
            last_layer_size = layer_size
        self.x2phi.append(nn.Linear(last_layer_size, self.x_phi_layers[-1], bias=True))
        self.f_x2phi = nn.Sequential(*self.x2phi).to(self.device)

        # input encoding layers
        self.xphi2h0 = nn.Linear(self.x_phi_layers[-1], self.d_layers[0], bias=True).to(self.device)

        if self.action_feedback:
            self.f_daphi2mu_q = nn.ModuleList()
            self.f_da2mu_p = nn.ModuleList()
            if isinstance(self.sig_scale, float):
                self.f_daphi2sig_q = lambda x: torch.tensor(self.sig_scale, dtype=torch.float32)
                self.f_da2sig_p = lambda x: torch.tensor(self.sig_scale, dtype=torch.float32)
            else:
                self.f_daphi2sig_q = nn.ModuleList()
                self.f_da2sig_p = nn.ModuleList()
        else:
            self.f_dphi2mu_q = nn.ModuleList()
            self.f_d2mu_p = nn.ModuleList()
            if isinstance(self.sig_scale, float):
                self.f_dphi2sig_q = lambda x: torch.tensor(self.sig_scale, dtype=torch.float32)
                self.f_d2sig_p = lambda x: torch.tensor(self.sig_scale, dtype=torch.float32)
            else:
                self.f_dphi2sig_q = nn.ModuleList()
                self.f_d2sig_p = nn.ModuleList()

        for lev in range(self.n_levels):

            if self.action_feedback:
                daphi2mu_q = nn.ModuleList()
                daphi2sig_q = nn.ModuleList()
                last_layer_size = self.d_layers[lev] + 1 + self.x_phi_layers[-1]  # 1是动作的维度
                for layer_size in self.posterior_layers:
                    daphi2mu_q.append(nn.Linear(last_layer_size, layer_size, bias=True))
                    daphi2mu_q.append(feedforward_actfun_rnn())
                    daphi2sig_q.append(nn.Linear(last_layer_size, layer_size, bias=True))
                    daphi2sig_q.append(feedforward_actfun_rnn())
                    last_layer_size = layer_size
                daphi2mu_q.append(nn.Linear(last_layer_size, self.z_layers[lev], bias=True))
                daphi2sig_q.append(nn.Linear(last_layer_size, self.z_layers[lev], bias=True))
                daphi2sig_q.append(nn.Softplus())
                self.f_daphi2mu_q.append(nn.Sequential(*daphi2mu_q)).to(self.device)
                if not isinstance(self.sig_scale, float):
                    self.f_daphi2sig_q.append(nn.Sequential(*daphi2sig_q)).to(self.device)

                da2mu_p = nn.ModuleList()
                da2sig_p = nn.ModuleList()
                last_layer_size = self.d_layers[lev] + 1      # 1是动作的维度
                for layer_size in self.prior_layers:
                    da2mu_p.append(nn.Linear(last_layer_size, layer_size, bias=True))
                    da2mu_p.append(feedforward_actfun_rnn())
                    da2sig_p.append(nn.Linear(last_layer_size, layer_size, bias=True))
                    da2sig_p.append(feedforward_actfun_rnn())
                    last_layer_size = layer_size
                da2mu_p.append(nn.Linear(last_layer_size, self.z_layers[lev], bias=True))
                da2sig_p.append(nn.Linear(last_layer_size, self.z_layers[lev], bias=True))
                da2sig_p.append(nn.Softplus())
                self.f_da2mu_p.append(nn.Sequential(*da2mu_p)).to(self.device)
                if not isinstance(self.sig_scale, float):
                    self.f_da2sig_p.append(nn.Sequential(*da2sig_p)).to(self.device)

            else:
                dphi2mu_q = nn.ModuleList()
                dphi2sig_q = nn.ModuleList()
                last_layer_size = self.d_layers[lev] + self.x_phi_layers[-1]
                for layer_size in self.posterior_layers:
                    dphi2mu_q.append(nn.Linear(last_layer_size, layer_size, bias=True))
                    dphi2mu_q.append(feedforward_actfun_rnn())
                    dphi2sig_q.append(nn.Linear(last_layer_size, layer_size, bias=True))
                    dphi2sig_q.append(feedforward_actfun_rnn())
                    last_layer_size = layer_size
                dphi2mu_q.append(nn.Linear(last_layer_size, self.z_layers[lev], bias=True))
                dphi2sig_q.append(nn.Linear(last_layer_size, self.z_layers[lev], bias=True))
                dphi2sig_q.append(nn.Softplus())
                self.f_dphi2mu_q.append(nn.Sequential(*dphi2mu_q)).to(self.device)
                if not isinstance(self.sig_scale, float):
                    self.f_dphi2sig_q.append(nn.Sequential(*dphi2sig_q)).to(self.device)

                d2mu_p = nn.ModuleList()
                d2sig_p = nn.ModuleList()
                last_layer_size = self.d_layers[lev]
                for layer_size in self.prior_layers:
                    d2mu_p.append(nn.Linear(last_layer_size, layer_size, bias=True))
                    d2mu_p.append(feedforward_actfun_rnn())
                    d2sig_p.append(nn.Linear(last_layer_size, layer_size, bias=True))
                    d2sig_p.append(feedforward_actfun_rnn())
                    last_layer_size = layer_size
                d2mu_p.append(nn.Linear(last_layer_size, self.z_layers[lev], bias=True))
                d2sig_p.append(nn.Linear(last_layer_size, self.z_layers[lev], bias=True))
                d2sig_p.append(nn.Softplus())
                self.f_d2mu_p.append(nn.Sequential(*d2mu_p)).to(self.device)
                if not isinstance(self.sig_scale, float):
                    self.f_d2sig_p.append(nn.Sequential(*d2sig_p)).to(self.device)

        # recurrent connections
        if self.rnn_type == 'mtrnn':
            self.z2h = nn.ModuleList()
            self.d2h = nn.ModuleDict()
            for l in range(self.n_levels):
                self.z2h.append(nn.Linear(self.z_layers[l], self.d_layers[l]))

                m = nn.Linear(d_layers[l], d_layers[l], bias=True)  # link from current level
                self.d2h["{}to{}".format(l, l)] = m
                if l > 0:  # not lowest level, link from one level lower
                    m = nn.Linear(d_layers[l - 1], d_layers[l], bias=True)
                    self.d2h["{}to{}".format(l - 1, l)] = m
                if l < self.n_levels - 1:  # not highest level, link from one level lower
                    m = nn.Linear(d_layers[l + 1], d_layers[l], bias=True)
                    self.d2h["{}to{}".format(l + 1, l)] = m

        elif self.rnn_type == 'mtlstm':
            self.rnn_levels = nn.ModuleList()
            for l in range(self.n_levels):
                if l == 0:
                    if self.n_levels == 1:
                        rnn_input_size = self.x_phi_layers[-1] + self.z_layers[l]
                    else:
                        rnn_input_size = self.x_phi_layers[-1] + self.d_layers[l + 1] + self.z_layers[l]

                elif l == self.n_levels - 1:
                    rnn_input_size = self.d_layers[l - 1] + self.z_layers[l]

                else:
                    rnn_input_size = self.d_layers[l - 1] + self.d_layers[l + 1] + self.z_layers[l]

                self.rnn_levels.append(nn.LSTMCell(rnn_input_size, self.d_layers[l])).to(self.device)

        else:
            raise ValueError("rnn_type must be 'mtrnn' or 'mtlstm'")

        # output decoding layers
        self.dz2mux = nn.ModuleList()
        self.dz2sigx = nn.ModuleList()

        last_layer_size = self.d_layers[0] + self.z_layers[0]
        for layer_size in self.decode_layers:
            self.dz2mux.append(nn.Linear(last_layer_size, layer_size, bias=True))
            self.dz2mux.append(feedforward_actfun_rnn())
            self.dz2sigx.append(nn.Linear(last_layer_size, layer_size, bias=True))
            self.dz2sigx.append(feedforward_actfun_rnn())
            last_layer_size = layer_size
        self.dz2mux.append(nn.Linear(last_layer_size, self.input_size, bias=True))
        self.f_dz2mux = nn.Sequential(*self.dz2mux).to(self.device)

        self.dz2sigx.append(nn.Linear(last_layer_size, self.input_size, bias=True))
        self.dz2sigx.append(nn.Softplus())
        if isinstance(self.sig_scale, float):
            self.f_dz2sigx = lambda x: torch.tensor(self.sig_scale, dtype=torch.float32)
        else:
            self.f_dz2sigx = nn.Sequential(*self.dz2sigx).to(self.device)

        # optimizer
        if optimizer == 'rmsprop':
            self.optimizer_st = torch.optim.RMSprop(self.parameters(), lr=lr_st, alpha=0.99)
        elif optimizer == 'adam':
            self.optimizer_st = torch.optim.Adam(self.parameters(), lr=lr_st)

    def rnn(self, prev_h_levels, prev_d_levels, new_z_levels, x_phi):
        new_h_levels = []
        new_d_levels = []

        if self.rnn_type == 'mtrnn':

            for l in range(self.n_levels):

                new_h = (1.0 - 1.0 / self.taus[l]) * prev_h_levels[l]

                new_h += (1.0 / self.taus[l]) * self.d2h["{}to{}".format(l, l)](prev_d_levels[l])
                if l > 0:
                    new_h += (1.0 / self.taus[l]) * self.d2h["{}to{}".format(l - 1, l)](prev_d_levels[l - 1])
                if l < self.n_levels - 1:
                    new_h += (1.0 / self.taus[l]) * self.d2h["{}to{}".format(l + 1, l)](prev_d_levels[l + 1])

                new_h += (1.0 / self.taus[l]) * self.z2h[l](new_z_levels[l])

                ## encode input
                if l == 0:
                    new_h += (1.0 / self.taus[l]) * self.xphi2h0(x_phi)

                new_h_levels.append(new_h)
                new_d_levels.append(torch.tanh(new_h))

        elif self.rnn_type == 'mtlstm':

            for l in range(self.n_levels):
                if l == 0:
                    if self.n_levels == 1:
                        rnn_input = x_phi
                    else:
                        rnn_input = torch.cat((x_phi, prev_d_levels[l + 1]), dim=-1)
                elif l == self.n_levels - 1:
                    rnn_input = prev_d_levels[l - 1]
                else:
                    rnn_input = torch.cat((prev_d_levels[l - 1], prev_d_levels[l + 1]), dim=-1)

                last = torch.cat((rnn_input, new_z_levels[l]), dim=-1)

                new_d, new_h = self.rnn_levels[l](last.to(self.device), (prev_d_levels[l].to(self.device), prev_h_levels[l].to(self.device)))

                # dilated LSTM
                mask_new = torch.rand_like(new_h, dtype=torch.float32) - 1 / self.taus[l]
                mask_new = (1.0 - torch.sign(mask_new)) / 2.0

                mask_old = torch.ones_like(new_h, dtype=torch.float32) - mask_new

                new_d = mask_new * new_d + mask_old * prev_d_levels[l].to(self.device)
                new_h = mask_new * new_h + mask_old * prev_h_levels[l].to(self.device)

                new_h_levels.append(new_h)
                new_d_levels.append(new_d)

        return new_h_levels, new_d_levels

    def sample_z(self, mu, sig):
        # Using reparameterization trick to sample from a gaussian
        if isinstance(sig, torch.Tensor):
            eps = Variable(torch.randn_like(mu))
        else:
            eps = torch.randn_like(mu)
        return mu + sig * eps

    def forward_generative(self, prev_h_levels, prev_d_levels, a_prev):
        # one-step generation

        # prior
        if self.action_feedback:
            a_prev = a_prev.view(prev_h_levels[0].size()[0], -1).to(self.device)
            mu_levels = [self.f_da2mu_p[l](torch.cat((prev_d_levels[l].to(self.device), a_prev), dim=-1)) for l in range(self.n_levels)]
            if isinstance(self.sig_scale, float):
                sig_levels = [torch.tensor(self.sig_scale, dtype=torch.float32) for l in range(self.n_levels)]
            else:
                sig_levels = [self.f_da2sig_p[l](torch.cat((prev_d_levels[l].to(self.device), a_prev), dim=-1)) for l in range(self.n_levels)]
        else:
            mu_levels = [self.f_d2mu_p[l](prev_d_levels[l].to(self.device)) for l in range(self.n_levels)]
            if isinstance(self.sig_scale, float):
                sig_levels = [torch.tensor(self.sig_scale, dtype=torch.float32) for l in range(self.n_levels)]
            else:
                sig_levels = [self.f_d2sig_p[l](prev_d_levels[l].to(self.device)) for l in range(self.n_levels)]

        new_z_p_levels = [self.sample_z(mu_levels[l], sig_levels[l]) for l in range(self.n_levels)]

        # pred x

        d0_prev = prev_d_levels[0]
        z0_new = new_z_p_levels[0]

        last = torch.cat((d0_prev.to(self.device), z0_new.to(self.device)), dim=-1)
        mux = self.f_dz2mux(last)

        last = torch.cat((d0_prev.to(self.device), z0_new.to(self.device)), dim=-1)
        sigx = self.f_dz2sigx(last) + SIG_MIN

        x_pred = self.sample_z(mux, sigx)

        # feature extraction
        x_phi = self.f_x2phi(x_pred)

        new_h_levels, new_d_levels = self.rnn(prev_h_levels, prev_d_levels, new_z_p_levels, x_phi)

        return x_pred, new_h_levels, new_d_levels, new_z_p_levels, mu_levels, sig_levels, mux, sigx

    def forward_inference(self, prev_h_levels, prev_d_levels, x_obs, a_prev_obs):

        # 对输入进行特征提取
        last = x_obs.view(prev_h_levels[0].size()[0], -1).to(self.device)
        x_phi = self.f_x2phi(last)

        if self.action_feedback:
            a_prev_obs = a_prev_obs.view(prev_h_levels[0].size()[0], -1).to(self.device)

        # posterior
        if self.action_feedback:
            mu_levels = [self.f_daphi2mu_q[l](torch.cat((prev_d_levels[l].to(self.device), a_prev_obs.float(), x_phi), dim=-1)) for l in
                         range(self.n_levels)]
            if isinstance(self.sig_scale, float):
                sig_levels = [torch.tensor(self.sig_scale, dtype=torch.float32) for l in range(self.n_levels)]
            else:
                sig_levels = [self.f_daphi2sig_q[l](torch.cat((prev_d_levels[l].to(self.device), a_prev_obs.float(), x_phi), dim=-1)) for l in
                              range(self.n_levels)]
        else:
            mu_levels = [self.f_dphi2mu_q[l](torch.cat((prev_d_levels[l].to(self.device), x_phi), dim=-1)) for l in
                         range(self.n_levels)]
            if isinstance(self.sig_scale, float):
                sig_levels = [torch.tensor(self.sig_scale, dtype=torch.float32) for l in range(self.n_levels)]
            else:
                sig_levels = [self.f_dphi2sig_q[l](torch.cat((prev_d_levels[l].to(self.device), x_phi), dim=-1)) for l in
                              range(self.n_levels)]

        new_z_q_levels = [self.sample_z(mu_levels[l], sig_levels[l]) for l in range(self.n_levels)]

        new_h_levels, new_d_levels = self.rnn(prev_h_levels, prev_d_levels, new_z_q_levels, x_phi)

        return new_h_levels, new_d_levels, new_z_q_levels, mu_levels, sig_levels

    def train_st(self, x_obs, a_obs, h_levels_0=None, d_levels_0=None, h_0_detach=True, validity=None, done_obs=None, seq_len=64):

        if not validity is None:
            v = validity.cpu().numpy().reshape([x_obs.size()[0], x_obs.size()[1]])
            stps = np.sum(v, axis=1)
            max_stp = int(np.max(stps))

            x_obs = x_obs[:, :max_stp]
            a_obs = a_obs[:, :max_stp]

            if not done_obs is None:
                done_obs = done_obs[:, :max_stp]

            validity = validity[:, :max_stp].reshape([x_obs.size()[0], x_obs.size()[1]])

        batch_size = x_obs.size()[0]

        if validity is None:  # no need for padding
            validity = torch.ones([x_obs.size()[0], x_obs.size()[1]], requires_grad=False)

        if h_levels_0 is None:
            h_levels_0 = self.init_hidden_zeros(batch_size=batch_size)
        elif isinstance(h_levels_0[0], np.ndarray):
                h_levels_0 = [torch.from_numpy(h_0) for h_0 in h_levels_0]

        if d_levels_0 is None:
            d_levels_0 = self.init_hidden_zeros(batch_size=batch_size)
        elif isinstance(d_levels_0[0], np.ndarray):
            d_levels_0 = [torch.from_numpy(d_0) for d_0 in d_levels_0]

        if h_0_detach:
            h_levels_init = [h_0.detach() for h_0 in h_levels_0]
            d_levels_init = [d_0.detach() for d_0 in d_levels_0]
            h_levels = h_levels_init
            d_levels = d_levels_init
        else:
            h_levels_init = [h_0 for h_0 in h_levels_0]
            d_levels_init = [d_0 for d_0 in d_levels_0]
            h_levels = h_levels_init
            d_levels = d_levels_init

        x_obs = x_obs.data
        a_obs = a_obs.data
        if not done_obs is None:
            done_obs = done_obs.data

        # sample minibatch of minibatch_size x seq_len
        stps_burnin = 100 #64
        x_sampled = torch.zeros([x_obs.size()[0], seq_len, x_obs.size()[-1]], dtype=torch.float32)
        a_sampled = torch.zeros([a_obs.size()[0], seq_len, a_obs.size()[-1]], dtype=torch.float32)
        v_sampled = torch.zeros([validity.size()[0], seq_len], dtype=torch.float32)

        for b in range(x_obs.size()[0]):
            v = validity.cpu().numpy().reshape([x_obs.size()[0], x_obs.size()[1]])
            stps = np.sum(v[b], axis=0).astype(int)
            start_index = np.random.randint(-seq_len + 1, stps - 1)

            for tmp, TMP in zip((x_sampled, a_sampled, v_sampled), (x_obs, a_obs, validity)):

                if start_index < 0 and start_index + seq_len > stps:
                    tmp[b, :stps] = TMP[b, :stps]

                elif start_index < 0:
                    tmp[b, :(start_index + seq_len)] = TMP[b, :(start_index + seq_len)]

                elif start_index + seq_len > stps:
                    tmp[b, :(stps - start_index)] = TMP[b, start_index:stps]

                else:
                    tmp[b] = TMP[b, start_index: (start_index + seq_len)]

            h_levels_b = [h_level[b:b+1] for h_level in h_levels]
            d_levels_b = [d_level[b:b+1] for d_level in d_levels]

            if start_index < 1:
                pass
            else:
                x_tmp = x_obs[b:b+1, max(0, start_index - stps_burnin):start_index]
                a_tmp = a_obs[b:b+1, max(0, start_index - stps_burnin):start_index]

                for t_burnin in range(x_tmp.size()[1]):

                    h_levels_b, d_levels_b, _, _, _ = self.forward_inference(h_levels_b, d_levels_b,
                                                                             x_tmp[:, t_burnin], a_tmp[:, t_burnin])

                for lev in range(self.n_levels):
                    h_levels[lev][b] = h_levels_b[lev][0].data
                    d_levels[lev][b] = d_levels_b[lev][0].data
        KL = 0

        h_series_levels = [[] for l in range(self.n_levels)]
        d_series_levels = [[] for l in range(self.n_levels)]
        z_p_series_levels = [[] for l in range(self.n_levels)]
        sig_p_series_levels = [[] for l in range(self.n_levels)]
        sig_q_series_levels = [[] for l in range(self.n_levels)]
        mu_p_series_levels = [[] for l in range(self.n_levels)]
        mu_q_series_levels = [[] for l in range(self.n_levels)]
        mux_pred_series = []
        sigx_pred_series = []

        for stp in range(seq_len):

            curr_x_obs = x_sampled[:, stp]
            prev_a_obs = a_sampled[:, stp]

            a_prev = prev_a_obs if self.action_feedback else None

            if not isinstance(self.sig_scale, float):
                x_pred, _, _, z_p_levels, mu_p_levels, sig_p_levels, mux_pred, sigx_pred = self.forward_generative(
                    h_levels, d_levels, a_prev)
                h_levels, d_levels, z_q_levels, mu_q_levels, sig_q_levels = self.forward_inference(
                    h_levels, d_levels, curr_x_obs, prev_a_obs)
            else:
                x_pred, _, _, z_p_levels, mu_p_levels, sig_p_levels, _, _ = self.forward_generative(
                    h_levels, d_levels, a_prev)
                h_levels, d_levels_new, z_q_levels, mu_q_levels, sig_q_levels = self.forward_inference(
                    h_levels, d_levels, curr_x_obs, prev_a_obs)

                last = torch.cat((d_levels[0], z_q_levels[0]), dim=-1)
                mux_pred = self.f_dz2mux(last)
                sigx_pred = self.f_dz2sigx(last) + SIG_MIN

                d_levels = d_levels_new

            # KL divergence term

            for l in range(self.n_levels):
                h_series_levels[l].append(h_levels[l])
                d_series_levels[l].append(d_levels[l])
                z_p_series_levels[l].append(z_p_levels[l])
                mu_p_series_levels[l].append(mu_p_levels[l])
                sig_p_series_levels[l].append(sig_p_levels[l])
                mu_q_series_levels[l].append(mu_q_levels[l])
                sig_q_series_levels[l].append(sig_q_levels[l])

            mux_pred_series.append(mux_pred)
            if not isinstance(self.sig_scale, float):
                sigx_pred_series.append(sigx_pred)

        if not isinstance(self.sig_scale, float):
            sig_p_tensor_levels = [torch.stack(sig_p_series_levels[l], dim=1) for l in range(self.n_levels)]
            sig_q_tensor_levels = [torch.stack(sig_q_series_levels[l], dim=1) for l in range(self.n_levels)]
            sigx_pred_tensor = torch.stack(sigx_pred_series, dim=1)

        mu_p_tensor_levels = [torch.stack(mu_p_series_levels[l], dim=1) for l in range(self.n_levels)]
        mu_q_tensor_levels = [torch.stack(mu_q_series_levels[l], dim=1) for l in range(self.n_levels)]
        mux_pred_tensor = torch.stack(mux_pred_series, dim=1)

        if not isinstance(self.sig_scale, float):
            for l in range(self.n_levels):
                KL += torch.mean(torch.mean(torch.log(sig_p_tensor_levels[l].to(self.device)) - torch.log(sig_q_tensor_levels[l].to(self.device))
                                 + ((mu_p_tensor_levels[l].to(self.device) - mu_q_tensor_levels[l].to(self.device)).pow(2) + sig_q_tensor_levels[l].to(self.device).pow(2))
                                 / (2.0 * sig_p_tensor_levels[l].to(self.device).pow(2)) - 0.5, dim=-1) * v_sampled.to(self.device)) / self.n_levels

            # log likelihood term
            Log = torch.mean(torch.mean(- torch.pow(mux_pred_tensor.to(self.device) - x_sampled.to(self.device), 2) / torch.pow(sigx_pred_tensor.to(self.device), 2) / 2
                             - torch.log(sigx_pred_tensor.to(self.device) * 2.5066), dim=-1) * v_sampled.to(self.device))
            elbo = - KL + Log
            loss = - elbo
        else:
            loss = torch.mean((mux_pred_tensor - x_sampled).pow(2))

        loss.to(self.device)
        self.optimizer_st.zero_grad()
        loss.backward()
        self.optimizer_st.step()

        return loss.cpu().item(), h_levels_init, d_levels_init

    def init_hidden_zeros(self, batch_size=1):

        h_levels = [torch.zeros((batch_size, d_size)) for d_size in self.d_layers]

        return h_levels


class VRDM(nn.Module):

    def __init__(self,
                 fim: VRM,
                 actor_lr=1e-3,
                 critic_lr=1e-2,
                 alpha_lr=1e-2,
                 gamma=0.99,
                 feedforward_actfun_sac=nn.ReLU,
                 beta_h='auto_1.0',
                 policy_layers=[256, 256],
                 value_layers=[256, 256],
                 tau=0.005,
                 use_expert=False,
                 source_model_path = [],
                 device="cpu"):

        super(VRDM, self).__init__()

        self.fim = fim
        self.gamma = gamma
        self.beta_h = beta_h
        self.policy_layers = policy_layers
        self.value_layers = value_layers
        self.tau = tau
        self.use_expert = use_expert
        self.source_model_path = source_model_path
        self.device = device

        self.input_size = self.fim.input_size
        self.action_size = self.fim.action_size
        self.a_prev = None
        self.include_obs = True
        self.target_entropy = -1  # 目标熵的大小

        # CUP 算法的两个超参数
        if self.use_expert:
            self.kl_weight = 30  # β1
            self.clip_thres = 3e-3  # β2

            self.modelList = []

            # 载入源策略
            for i in range(len(self.source_model_path)):
                model = torch.load(self.source_model_path[i])
                self.modelList.append(model)

        # d_layers各层的神经元数
        self.d_layers = []
        for lev in range(self.fim.n_levels):
            self.d_layers.append(self.fim.d_layers[lev])
        # d_layers的层数
        self.n_levels = len(self.d_layers)

        self.forward_inference_fim = self.fim.forward_inference

        self.h_levels = self.init_hidden_zeros(batch_size=1)
        self.d_levels = self.init_hidden_zeros(batch_size=1)

        if isinstance(self.beta_h, str) and self.beta_h.startswith('auto'):
            init_value = 1.0
            if '_' in self.beta_h:
                init_value = float(self.beta_h.split('_')[1])
                assert init_value > 0., "The initial value of beta_h must be greater than 0"
            # 使用其log值能够使训练更稳定
            self.log_beta_h = torch.tensor(np.log(init_value).astype(np.float32), requires_grad=True)
            self.optimizer_e = torch.optim.Adam([self.log_beta_h], lr=alpha_lr)
            
        # policies network
        self.d2a = nn.ModuleList()
        last_layer_size = self.d_layers[0] if not self.include_obs else self.d_layers[0] + self.input_size
        for layer_size in self.policy_layers:
            self.d2a.append(nn.Linear(last_layer_size, layer_size, bias=True))
            self.d2a.append(feedforward_actfun_sac())
            last_layer_size = layer_size
        self.d2a.append(nn.Linear(last_layer_size, self.action_size, bias=True))
        self.d2a.append(nn.Softmax(dim=-1))
        self.f_d2a = nn.Sequential(*self.d2a).to(self.device)

        # soft Q network1
        self.d2q1 = nn.ModuleList()
        last_layer_size = self.d_layers[0] if not self.include_obs else self.d_layers[0] + self.input_size
        for layer_size in self.value_layers:
            self.d2q1.append(nn.Linear(last_layer_size, layer_size, bias=True))
            self.d2q1.append(feedforward_actfun_sac())
            last_layer_size = layer_size
        self.d2q1.append(nn.Linear(last_layer_size, self.action_size, bias=True))
        self.f_d2q1 = nn.Sequential(*self.d2q1).to(self.device)

        # soft Q network2
        self.d2q2 = nn.ModuleList()
        last_layer_size = self.d_layers[0] if not self.include_obs else self.d_layers[0] + self.input_size
        for layer_size in self.value_layers:
            self.d2q2.append(nn.Linear(last_layer_size, layer_size, bias=True))
            self.d2q2.append(feedforward_actfun_sac())
            last_layer_size = layer_size
        self.d2q2.append(nn.Linear(last_layer_size, self.action_size, bias=True))
        self.f_d2q2 = nn.Sequential(*self.d2q2).to(self.device)

        # soft Q network1 target
        self.d2q1_target = nn.ModuleList()
        last_layer_size = self.d_layers[0] if not self.include_obs else self.d_layers[0] + self.input_size
        for layer_size in self.value_layers:
            self.d2q1_target.append(nn.Linear(last_layer_size, layer_size, bias=True))
            self.d2q1_target.append(feedforward_actfun_sac())
            last_layer_size = layer_size
        self.d2q1_target.append(nn.Linear(last_layer_size, self.action_size, bias=True))
        self.f_d2q1_target = nn.Sequential(*self.d2q1_target).to(self.device)

        # soft Q network2 target
        self.d2q2_target = nn.ModuleList()
        last_layer_size = self.d_layers[0] if not self.include_obs else self.d_layers[0] + self.input_size
        for layer_size in self.value_layers:
            self.d2q2_target.append(nn.Linear(last_layer_size, layer_size, bias=True))
            self.d2q2_target.append(feedforward_actfun_sac())
            last_layer_size = layer_size
        self.d2q2_target.append(nn.Linear(last_layer_size, self.action_size, bias=True))
        self.f_d2q2_target = nn.Sequential(*self.d2q2_target).to(self.device)

        self.f_d2q1_target.load_state_dict(self.f_d2q1.state_dict())
        self.f_d2q2_target.load_state_dict(self.f_d2q2.state_dict())

        self.optimizer_a = torch.optim.Adam(self.f_d2a.parameters(), lr=actor_lr)
        self.optimizer_v1 = torch.optim.Adam(self.f_d2q1.parameters(), lr=critic_lr)
        self.optimizer_v2 = torch.optim.Adam(self.f_d2q2.parameters(), lr=critic_lr)

    def sample_action(self, d0_prev, x_prev):
        if not self.include_obs:
            s = d0_prev.to(self.device)
        else:
            s = torch.cat((d0_prev, x_prev), dim=-1).to(self.device)

        probs = self.f_d2a(s)
        action_dist =dis.Categorical(probs)
        action = action_dist.sample()
        return action.detach()

    def preprocess_sac(self, x_obs, r_obs, a_obs, d_obs=None, v_obs=None, seq_len=64):

        if not v_obs is None:
            v = v_obs.cpu().numpy().reshape([x_obs.size()[0], x_obs.size()[1]])
            stps = np.sum(v, axis=1)
            max_stp = int(np.max(stps))

            x_obs = x_obs[:, :max_stp]
            a_obs = a_obs[:, :max_stp]
            r_obs = r_obs[:, :max_stp]
            d_obs = d_obs[:, :max_stp]
            v_obs = v_obs[:, :max_stp]

        batch_size = x_obs.size()[0]
        start_indices = np.zeros(x_obs.size()[0], dtype=int)

        for b in range(x_obs.size()[0]):
            v = v_obs.cpu().numpy().reshape([x_obs.size()[0], x_obs.size()[1]])
            stps = np.sum(v[b], axis=0).astype(int)
            start_indices[b] = np.random.randint(-seq_len + 1, stps - 1)

        x_obs = x_obs.data
        a_obs = a_obs.data
        r_obs = r_obs.data
        d_obs = d_obs.data
        v_obs = v_obs.data

        # initialize hidden states
        h_levels_0 = self.init_hidden_zeros(batch_size=batch_size)
        d_levels_0 = self.init_hidden_zeros(batch_size=batch_size)

        h_levels = [h_0.detach() for h_0 in h_levels_0]
        d_levels = [d_0.detach() for d_0 in d_levels_0]

        h_levels_fim = []
        d_levels_fim = []

        for lev in range(self.n_levels):
            h_levels_fim.append(h_levels[lev][:, :self.fim.d_layers[lev]])
            d_levels_fim.append(d_levels[lev][:, :self.fim.d_layers[lev]])

        # ========================= FIM =========================

        # h_series_levels = [[] for l in range(self.n_levels)]
        d_series_levels_fim = [[] for l in range(self.n_levels)]

        stps_burnin = 100 #64

        x_sampled = torch.zeros([x_obs.size()[0], seq_len + 1, x_obs.size()[-1]], dtype=torch.float32)  # +1 for SP
        a_sampled = torch.zeros([a_obs.size()[0], seq_len + 1, a_obs.size()[-1]], dtype=torch.float32)

        for b in range(x_obs.size()[0]):
            v = v_obs.cpu().numpy().reshape([x_obs.size()[0], x_obs.size()[1]])
            stps = np.sum(v[b], axis=0).astype(int)
            start_index = start_indices[b]

            for tmp, TMP in zip((x_sampled, a_sampled), (x_obs, a_obs)):

                if start_index < 0 and start_index + seq_len + 1 > stps:
                    tmp[b, :stps] = TMP[b, :stps]

                elif start_index < 0:
                    tmp[b, :(start_index + seq_len + 1)] = TMP[b, :(start_index + seq_len + 1)]

                elif start_index + seq_len + 1 > stps:
                    tmp[b, :(stps - start_index)] = TMP[b, start_index:stps]

                else:
                    tmp[b] = TMP[b, start_index: (start_index + seq_len + 1)]

            h_levels_b_fim = [h_level[b:b + 1] for h_level in h_levels_fim]
            d_levels_b_fim = [d_level[b:b + 1] for d_level in d_levels_fim]

            if start_index < 1:
                pass
            else:
                x_tmp = x_obs[b:b + 1, max(0, start_index - stps_burnin):start_index]
                a_tmp = a_obs[b:b + 1, max(0, start_index - stps_burnin):start_index]

                for t_burnin in range(x_tmp.size()[0]):
                    x_tmp_t = x_tmp[:, t_burnin]
                    a_tmp_t = a_tmp[:, t_burnin] if self.fim.action_feedback else None
                    h_levels_b_fim, d_levels_b_fim, _, _, _ = self.forward_inference_fim(h_levels_b_fim, d_levels_b_fim,
                                                                                         x_tmp_t, a_tmp_t)

                for lev in range(self.n_levels):
                    h_levels_fim[lev][b] = h_levels_b_fim[lev][0].data
                    d_levels_fim[lev][b] = d_levels_b_fim[lev][0].data

        for stp in range(seq_len + 1):
            curr_x_obs = x_sampled[:, stp]
            prev_a_obs = a_sampled[:, stp] if self.fim.action_feedback else None
            h_levels_fim, d_levels_fim, _, _, _ = self.forward_inference_fim(h_levels_fim, d_levels_fim, curr_x_obs, prev_a_obs)

            for l in range(self.n_levels):
                d_series_levels_fim[l].append(d_levels_fim[l].detach())


        d_low_tensor_fim = torch.stack(d_series_levels_fim[0], dim=1).detach().data

        S_sampled_fim = d_low_tensor_fim[:, :-1, :]
        SP_sampled_fim = d_low_tensor_fim[:, 1:, :]

        # ========================= END - FIM =========================

        if self.include_obs:
            S_sampled = torch.cat((S_sampled_fim, x_sampled[:, :-1, :].to(self.device)), dim=-1)
            SP_sampled = torch.cat((SP_sampled_fim, x_sampled[:, 1:, :].to(self.device)), dim=-1)
        else:
            S_sampled = S_sampled_fim
            SP_sampled = SP_sampled_fim

        A = a_obs
        R = r_obs

        if d_obs is None:
            D = torch.zeros_like(R, dtype=torch.float32)
        else:
            D = d_obs

        if v_obs is None:  # no need for padding
            V = torch.ones_like(R, requires_grad=False, dtype=torch.float32)
        else:
            V = v_obs

        A_sampled = torch.zeros([A.size()[0], seq_len + 1, 1], dtype=torch.float32)
        D_sampled = torch.zeros([D.size()[0], seq_len + 1, 1], dtype=torch.float32)
        R_sampled = torch.zeros([R.size()[0], seq_len + 1, 1], dtype=torch.float32)
        V_sampled = torch.zeros([V.size()[0], seq_len + 1, 1], dtype=torch.float32)

        for b in range(A.size()[0]):
            v = v_obs.cpu().numpy().reshape([A.size()[0], A.size()[1]])
            stps = np.sum(v[b], axis=0).astype(int)
            start_index = start_indices[b]

            # sampled_indices = np.arange(start_index, start_index + seq_len)

            for tmp, TMP in zip((A_sampled, D_sampled, R_sampled, V_sampled),
                                (A, D, R, V)):

                if start_index < 0 and start_index + seq_len + 1 > stps:
                    tmp[b, :stps] = TMP[b, :stps]

                elif start_index < 0:
                    tmp[b, :(start_index + seq_len + 1)] = TMP[b, :(start_index + seq_len + 1)]

                elif start_index + seq_len + 1 > stps:
                    tmp[b, :(stps - start_index)] = TMP[b, start_index:stps]

                else:
                    tmp[b] = TMP[b, start_index: (start_index + seq_len + 1)]

        R_sampled = R_sampled[:, :-1, :].data
        A_sampled = A_sampled[:, :-1, :].data
        D_sampled = D_sampled[:, :-1, :].data
        V_sampled = V_sampled[:, 1:, :].data

        return S_sampled, SP_sampled, A_sampled, R_sampled, D_sampled, V_sampled

    def train_rl_sac_(self, S_sampled, SP_sampled, A_sampled, R_sampled, D_sampled):
        
        states = S_sampled                      # 当前的观察
        next_states = SP_sampled                # 下一时刻的观察
        actions = A_sampled.to(self.device)     # 当前的 action
        rewards = R_sampled.to(self.device)     # 当前的 reward
        dones = D_sampled.to(self.device)       # 当前的 done

        # 更新 critic network
        td_target = self.calc_target(rewards, next_states, dones)

        critic_1_q_values = self.f_d2q1(states).gather(-1, actions.long())
        critic_1_loss = torch.mean(F.mse_loss(critic_1_q_values, td_target.detach())).to(self.device)

        critic_2_q_values = self.f_d2q2(states).gather(-1, actions.long())
        critic_2_loss = torch.mean(F.mse_loss(critic_2_q_values, td_target.detach())).to(self.device)

        self.optimizer_v1.zero_grad()
        critic_1_loss.backward()
        self.optimizer_v1.step()

        self.optimizer_v2.zero_grad()
        critic_2_loss.backward()
        self.optimizer_v2.step()

        # 更新 actor network
        probs = self.f_d2a(states)
        log_probs = torch.log(probs + 1e-8)

        entropy = -torch.sum(probs * log_probs, dim=-1, keepdim=True)
        q1_value = self.f_d2q1(states)
        q2_value = self.f_d2q2(states)
        min_qvalue = torch.sum(probs * torch.min(q1_value, q2_value), dim=-1, keepdim=True)

        actor_loss = torch.mean(-self.log_beta_h.exp() * entropy.cpu() - min_qvalue.cpu()).to(self.device)

        # 使用CUP方法从 source policy 学习
        if self.use_expert:
            # 计算当前policy的价值
            target_Q1 = self.f_d2q1_target(states)
            target_Q2 = self.f_d2q2_target(states)

            Q = torch.sum(probs * torch.max(target_Q1, target_Q2), dim=-1, keepdim=True).detach()
            Q = Q - self.log_beta_h.exp().detach() * torch.sum(probs * log_probs, -1, keepdim=True).detach()

            Q_e_all = [Q]
            kld_all = [None]

            for i in range(len(self.modelList)):
                prob = self.modelList[i].f_d2a(states)
                log_prob = torch.log(prob + 1e-8)

                Q_e = torch.sum(prob * torch.max(target_Q1, target_Q2), dim=-1, keepdim=True).detach()
                Q_e = Q_e - self.log_beta_h.exp().detach() * torch.sum(prob * log_prob, -1, keepdim=True).detach()
                Q_e_all.append(Q_e)

                kld_tmp = self.kl_divergence(probs, log_probs, log_prob)
                kld_all.append(kld_tmp)

            kld_all[0] = torch.zeros_like(kld_all[-1])

            Q_e_all = torch.cat(Q_e_all, -1)
            kld_all = torch.cat(kld_all, -1)

            num_candidates = Q_e_all.shape[-1]

            _, y = torch.max(Q_e_all, -1)
            weight = F.one_hot(y, num_classes=num_candidates)

            Q_weight = Q_e_all.detach() - Q_e_all[:, :, 0:1].detach()
            Q_weight = torch.min(Q_weight, self.clip_thres * abs(Q_e_all[:, :, 0:1]))
            Q_weight = Q_weight.detach()

            kld_loss = torch.mean(kld_all * weight * Q_weight)
            kld_loss = self.kl_weight * kld_loss  # β1 * kl_loss
            actor_loss += kld_loss

        
        self.optimizer_a.zero_grad()
        actor_loss.backward()
        self.optimizer_a.step()

        # 更新entropy coefficient
        beta_h_loss = torch.mean(
            (entropy.cpu() - self.target_entropy).detach() * self.log_beta_h).to(self.device)
        self.optimizer_e.zero_grad()
        beta_h_loss.backward()
        self.optimizer_e.step()

        self.soft_update(self.f_d2q1, self.f_d2q1_target)
        self.soft_update(self.f_d2q2, self.f_d2q2_target)

    def init_hidden_zeros(self, batch_size=1):

        h_levels = [torch.zeros((batch_size, d_size)) for d_size in self.d_layers]

        return h_levels

    def detach_states(self, states):
        states = [s.detach() for s in states]
        return states

    def init_episode(self, x_0=None, h_levels_0=None, d_levels_0=None):
        if h_levels_0 is None:
            self.h_levels = self.init_hidden_zeros(batch_size=1)
        else:
            self.h_levels = [torch.from_numpy(h0) for h0 in h_levels_0]

        if d_levels_0 is None:
            self.d_levels = self.init_hidden_zeros(batch_size=1)
        else:
            self.d_levels = [torch.from_numpy(d0) for d0 in d_levels_0]

        if x_0 is None:
            x_obs_0 = None
        else:
            x_obs_0 = torch.from_numpy(x_0).view(1, -1)

        a = self.sample_action(self.d_levels[0], x_obs_0)

        self.a_prev = a

        return a.cpu().numpy()

    def select(self, s, r_prev):
        r_prev = np.array([r_prev]).reshape([-1]).astype(np.float32)
        s = np.array(s).reshape([-1]).astype(np.float32)
        #x_obs = torch.cat((torch.from_numpy(s), torch.from_numpy(r_prev))).view([1, -1]).to(self.device)
        x_obs = torch.from_numpy(s).view([1, -1]).to(self.device)

        self.h_levels_fim = []
        self.d_levels_fim = []

        for lev in range(self.fim.n_levels):
            self.h_levels_fim.append(self.h_levels[lev][:, :self.fim.d_layers[lev]])
            self.d_levels_fim.append(self.d_levels[lev][:, :self.fim.d_layers[lev]])

        self.h_levels_fim, self.d_levels_fim, _, _, _ = self.forward_inference_fim(self.h_levels_fim, self.d_levels_fim,
                                                                                   x_obs, self.a_prev)
        for lev in range(self.n_levels):
            self.h_levels[lev] = self.h_levels_fim[lev]
            self.d_levels[lev] = self.d_levels_fim[lev]

        a = self.sample_action(self.d_levels[0], x_obs)

        self.a_prev = a

        return a.cpu().numpy()

    def learn_st(self, train_fim:bool, SP, A, R, D=None, V=None, H0=None, D0=None,
                 times=1, minibatch_size=4, seq_len=64):  # learning from the data of this episode

        if D is None:
            D = np.zeros_like(R, dtype=np.float32)
        if V is None:
            V = np.ones_like(R, dtype=np.float32)

        for xt in range(times):
            weights = np.sum(V, axis=-1) + 2 * seq_len - 2
            e_samples = np.random.choice(SP.shape[0], minibatch_size, p=weights / weights.sum())

            sp = SP[e_samples]
            a = A[e_samples]
            r = R[e_samples]
            d = D[e_samples]
            v = V[e_samples]

            if not H0 is None:
                h0 = [hl[e_samples] for hl in H0]
            else:
                h0 = None

            if not D0 is None:
                d0 = [dl[e_samples] for dl in D0]
            else:
                d0 = None

            r_obs = torch.from_numpy(r.reshape([r.shape[0], r.shape[1], 1]))
            #x_obs = torch.cat((torch.from_numpy(sp), r_obs), dim=-1)
            x_obs = torch.from_numpy(sp)

            a_obs = torch.from_numpy(a.reshape([r.shape[0], r.shape[1], 1]))

            d_obs = torch.from_numpy(d.reshape([r.shape[0], r.shape[1], 1]))

            v_obs = torch.from_numpy(v.reshape([r.shape[0], r.shape[1], 1]))

            if train_fim:
                loss, h_levels_init, d_levels_init = self.fim.train_st(x_obs, a_obs, h_levels_0=h0, validity=v_obs,
                                                                      d_levels_0=d0, h_0_detach=False, done_obs=d_obs,
                                                                      seq_len=seq_len)

        if not H0 is None:
            for l in range(len(H0)):
                H0[l][e_samples, :] = h_levels_init[l].cpu().detach().numpy()
                D0[l][e_samples, :] = d_levels_init[l].cpu().detach().numpy()

        return H0, D0, loss

    def learn_rl_sac(self, SP, A, R, D=None, V=None, H0=None, D0=None, times=1, minibatch_size=4, seq_len=64):

        if D is None:
            D = np.zeros_like(R, dtype=np.float32)
        if V is None:
            V = np.ones_like(R, dtype=np.float32)

        for xt in range(times):
            weights = np.sum(V, axis=-1) + 2 * seq_len - 2
            e_samples = np.random.choice(SP.shape[0], minibatch_size, p=weights / weights.sum())

            sp = SP[e_samples]
            a = A[e_samples]
            r = R[e_samples]
            d = D[e_samples]
            v = V[e_samples]

            if not H0 is None:
                h0 = [hl[e_samples] for hl in H0]
            else:
                h0 = None

            if not D0 is None:
                d0 = [dl[e_samples] for dl in D0]
            else:
                d0 = None

            r_obs = torch.from_numpy(r.reshape([r.shape[0], r.shape[1], 1]))
            #x_obs = torch.cat((torch.from_numpy(sp), r_obs), dim=-1)
            x_obs = torch.from_numpy(sp)

            a_obs = torch.from_numpy(a)
            d_obs = torch.from_numpy(d.reshape([r.shape[0], r.shape[1], 1]))
            v_obs = torch.from_numpy(v.reshape([r.shape[0], r.shape[1], 1]))

            S_sampled, SP_sampled, A_sampled, R_sampled, D_sampled, V_sampled \
                = self.preprocess_sac(x_obs, r_obs, a_obs, d_obs=d_obs, v_obs=v_obs, seq_len=seq_len)

            self.train_rl_sac_(S_sampled=S_sampled, SP_sampled=SP_sampled, A_sampled=A_sampled, R_sampled=R_sampled, D_sampled=D_sampled)

    # 计算目标Q值,直接用策略网络的输出概率进行期望计算
    def calc_target(self, rewards, next_states, dones):
        next_probs = self.f_d2a(next_states)
        next_log_probs = torch.log(next_probs + 1e-8)

        entropy = -torch.sum(next_probs * next_log_probs, dim=-1, keepdim=True)

        qf1_next_target = self.f_d2q1_target(next_states)
        qf2_next_target = self.f_d2q2_target(next_states)

        min_qvalue = torch.sum(next_probs * torch.min(qf1_next_target, qf2_next_target), dim=-1, keepdim=True)
        min_qf_next_target = min_qvalue + self.log_beta_h.exp() * entropy

        td_target = rewards + self.gamma * min_qf_next_target * (1 - dones)
        return td_target

    # 更新target network参数
    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def kl_divergence(self, p, log_p, log_q):

        temp1 = torch.sum(p * log_p, dim=-1, keepdim=True)
        temp2 = torch.sum(p * log_q, dim=-1, keepdim=True)
        kl_divergence = temp1 - temp2

        return kl_divergence
