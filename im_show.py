import cv2
import cv2 as cv
import os
import shutil

from regex import F
path = "./choose/outdoor_day_normal/" #"./hyz/"
save_path = "./choose/"
os.makedirs(save_path, exist_ok=True)
# folder_path1 = os.getcwd()
# print(folder_path1)
num = 0
# for files in os.listdir(path):
for root, _, files in os.walk(path):
    # print(files)
    files.sort()
    # print(files)
    # print(files[38948])
    begin = 0
    length=len(files)
    while begin < length: 
        subname = files[begin]
        # print(subname)
        img_path = os.path.join(root, subname)
        img = cv.imread(img_path)
        # print(img.shape)
        # print(type(img))
        cv.namedWindow('IMG', 0)
        cv.resizeWindow("IMG", 520, 520)
        ##################坑全屏################
        # out_win = "output_style_full_screen"
        # cv2.namedWindow(out_win, cv2.WINDOW_NORMAL)
        # cv2.setWindowProperty(out_win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv.imshow("IMG", img)
        #--------键盘控制视频---------------
        #读取键盘值
        key = cv2.waitKey(0) & 0xff
        print("length:" + str(length) + "           begin:" + str(begin) + "        path:" + img_path)
        #设置按#键移动保存
        #设置空格按下时向后
        if key == ord(" "):
            begin +=1
            continue
            # cv2.waitKey(0)
        #设置B按下时向前b
        elif key == ord("b"):
            begin -=1
            continue
        #设置Q按下时退出
        elif key == ord("q"):
            print(subname)
            break
        elif key == ord("n"):
            save_path_odn = os.path.join(save_path, "outdoor_day_normal/")
            os.makedirs(save_path_odn, exist_ok=True)
            shutil.move(img_path, save_path_odn)
            num = num + 1
        elif key == ord("o"):
            save_path_odo = os.path.join(save_path, "outdoor_day_obssim/")
            # print("save_path_odo:", save_path_odo)
            os.makedirs(save_path_odo, exist_ok=True)
            shutil.move(img_path, save_path_odo)
            num = num + 1
        elif key == ord("t"):
            save_path_odt = os.path.join(save_path, "outdoor_day_thinobj/")
            os.makedirs(save_path_odt, exist_ok=True)
            shutil.move(img_path, save_path_odt)
            num = num + 1   
        elif key == ord("l"):
            save_path_onl = os.path.join(save_path, "outdoor_night_light/")
            os.makedirs(save_path_onl, exist_ok=True)
            shutil.move(img_path, save_path_onl)
            num = num + 1
        
        begin +=1
        # cv.waitKey(100)
print("共复制%s张图到"%(num) + save_path)
cv.destroyAllWindows()
