import sys
sys.path.append("/semantic-segmentation-pytorch")
import json
import os
import numpy as np
import re
import PIL.Image
import PIL.ImageDraw
import cv2
from numpy import dtype
from PIL import Image
from mit_semseg.utils import colorEncode
from scipy.io import loadmat

colors = loadmat('/data/color8.mat')['colors']

json_path = "du_json/"
img_path = "/du/"
gt_path = "/du_gt/"
pre_path = "/du_pre/"
os.makedirs(gt_path, exist_ok=True)
os.makedirs(pre_path, exist_ok=True)

def draw_to_mask(segimg, points, shape_type="polygon", line_width=10):
    h, w = segimg.shape[0], segimg.shape[1]
    mask = np.zeros((h, w), np.uint8)
    mask = PIL.Image.fromarray(mask)
    draw = PIL.ImageDraw.Draw(mask)

    xy = [tuple(point) for point in points]

    if shape_type == 'circle':
        assert len(xy) == 2, 'Shape of shape_type=circle must have 2 points'
        (cx, cy), (px, py) = xy
        d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)

    elif shape_type == 'linestrip':
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == 'polygon':
        assert len(xy) > 2, 'Polygon must have points more than 2'
        draw.polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask


labels_id = {"road": 1, "ground_lock":2, "cone_barrel":3}
for js in os.listdir(json_path):
    imgName = js.split(".")[0]
    for img in os.listdir(img_path):
        # if img.startswith("NOR_20220111_113258_M_00001_0055_F"):
        imgPath = os.path.join(img_path, img)
        if img.startswith(imgName):
            road_points = []
            ground_points = []
            cone_points = []
            jsonFile_path = os.path.join(json_path, js)
            print(jsonFile_path)
            f = open(jsonFile_path, 'r')
            content = f.read()
            j = json.loads(content)
            h, w  = j["height"], j["width"]
            # print(w,h)
            segImg = np.zeros((h,w), dtype=np.uint8)
            if 'road' in j.keys():
                label = "road"
                road_value = j["road"]
                # print("type(road_value): ",type(road_value))
                # print("len(road_value):", len(road_value))
                for r in road_value:
                    r_points = r["points"]
                    # print(len(r_points))
                    # print(type(r_points))
                    road_points = np.array(r_points).reshape(int(len(r_points)/2),-1)              
                    # x = r_points[0]
                    # y = r_points[1]
                    # # print(x, y)
                    # xy=[x,y]
                    # road_points.append(xy)
                    # i = 0
                    # for p in r_points:
                    #     if i % 2 == 0:
                    #         x_tmp = x
                    #         x = p
                    #         i = i + 1
                    #     else:
                    #         y_tmp = y
                    #         y = p
                    #         i = i + 1
                    #     if x_tmp != x and y_tmp != y:
                    #         x_tmp, y_tmp = x, y
                    #         xy = [float(x), float(y)]
                    #         road_points.append(xy)
                    # # print(type(road_points))
                    # print(road_points)
                    mask = draw_to_mask(segImg, road_points)
                    segImg[mask] = labels_id[label]
            if 'ground_lock' in j.keys():
                label = "ground_lock"
                ground_value = j["ground_lock"]
                # print("type(ground_value): ",type(ground_value))
                # print("len(ground_value):", len(ground_value))
                for gv_list in ground_value:
                    ground_points = []
                    for g in gv_list:
                        g_points = g["points"]
                        ground_points = np.array(g_points).reshape(int(len(g_points)/2),-1)
                        # print(len(r_points))
                        # x, y = g_points[0], g_points[1]
                        # xy=[x,y]
                        # ground_points.append(xy)
                        # i = 0
                        # for p in g_points:
                        #     if i % 2 == 0:
                        #         x_tmp = x
                        #         x = p
                        #         i = i + 1
                        #     else:
                        #         y_tmp = y
                        #         y = p
                        #         i = i + 1
                        #     if x_tmp != x and y_tmp != y:
                        #         x_tmp, y_tmp = x, y
                        #         xy = [float(x), float(y)]
                        #         ground_points.append(xy)
                        mask = draw_to_mask(segImg, ground_points)
                        segImg[mask] = labels_id[label]
            if 'cone_barrel' in j.keys():
                label = "cone_barrel"
                cone_value = j["cone_barrel"]
                # print("type(cone_value): ",type(cone_value))
                # print("len(cone_value):", len(cone_value))
                for cv_list in cone_value:
                    cone_points = []
                    for c in cv_list:
                        c_points = c["points"]
                        # print(len(r_points))
                        cone_points = np.array(c_points).reshape(int(len(c_points)/2),-1)
                        # x, y = c_points[0], c_points[1]
                        # xy=[x,y]
                        # cone_points.append(xy)
                        # i = 0
                        # for p in c_points:
                        #     if i % 2 == 0:
                        #         x_tmp = x
                        #         x = p
                        #         i = i + 1
                        #     else:
                        #         y_tmp = y
                        #         y = p
                        #         i = i + 1
                        #     if x_tmp != x and y_tmp != y:
                        #         x_tmp, y_tmp = x, y
                        #         xy = [float(x), float(y)]
                        #         cone_points.append(xy)
                        # print(cone_points)
                        mask = draw_to_mask(segImg, cone_points)
                        segImg[mask] = labels_id[label]
            gt_name = img.replace("jpg", "png")
            ###实例图
            # Image.fromarray(segImg).save(os.path.join(gt_path, gt_name))
            ###预测图
            img = cv2.imread(imgPath)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            pred_color = colorEncode(segImg, colors)
            im_vis = cv2.addWeighted(img, 0.6, pred_color, 0.4, 0)
            Image.fromarray(im_vis).save(os.path.join(pre_path, gt_name))
