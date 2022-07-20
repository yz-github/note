from xml.dom.minidom import Element
import xml.etree.ElementTree as ET
import os
# from numpy import *
import operator
from lxml import etree
import json
import numpy as np
import cv2

xml_file_path = "/annotations_original_amout30768_0603_sortId_replaced.xml"
#要加入的xml
add_xml_file_path = "/annotations_0603szxa.xml"

del_xml_file_path = "/annotations_original_amout40395_0715.xml"

sort_xml_file_path = "/annotations_original_amout40125_0715.xml"

replace_xml_file_path = "/annotations_original_amout14160_0614.xml"

json_root_path = "/0605/"

compare_del_xml_file_path = "/annotations_original_amout40125_0715_sortId.xml"

compare_add_xml_file_path = "/annotations_original_amout40117_0715_.xml"

#获取最大id
def find_maxId(root):
    list = []
    for parent_node in root.iter('image'):
        id = int(parent_node.get("id"))
        list.append(id)
    max_index, max_id = max(enumerate(list), key=operator.itemgetter(1))
    return max_id

def sum_node(xml_file_path, add_xml_file_path):
    file = open(xml_file_path, 'rb')
    tree = ET.parse(file)
    root = tree.getroot()
    # print("root:", root)
    add_file = open(add_xml_file_path, 'rb')
    add_tree = ET.parse(add_file)
    add_root = add_tree.getroot()
    num = 0
    #原始id终点 最大id
    id_oriMax = find_maxId(root)  
    print("id_oriMax:", id_oriMax)
    for add_parent_node in add_root.iter('image'):
        #print("root:", root.getchildren())
        #print("add_root:", add_root.getchildren())
        #print("add_parent_node:", add_parent_node)    
        id = int(add_parent_node.get("id"))
        # print(type(add_parent_node.get("id")))
        # print(id)
        id = id + id_oriMax + 1
        add_parent_node.set("id", str(id))
        # id = int(add_parent_node.get("id"))
        # print("#####", id)
        # print(add_parent_node.get('name'))
        # print(len(root.getchildren()))
        root.append(add_parent_node)
        # print("加后", len(root.getchildren()))
        num = num + 1
    
    # print(len(root.getchildren()))
    print("共加入%s张图"%num)
    sum = id_oriMax + 1 + num
    Topath = "/mnt/data/semantic-segmentation-pytorch/annotations/sum/"
    os.makedirs(Topath, exist_ok = True)
    tree.write(Topath + "annotations_original_amout%s_0603.xml"%(sum), encoding='utf-8')
    print("write xml to " + Topath + "annotations_original_amout%s_0603.xml"%(sum))

def sort_id(sort_xml_file_path):
    file = open(sort_xml_file_path, 'rb')
    tree = ET.parse(file)
    root = tree.getroot()
    changeId_num = 0
    file_name = os.path.split(sort_xml_file_path)[1].split(".")[0]
    # print(file_name)

    ####id排序
    # id_tmp = -1
    # for parent_node in root.iter('image'): 
    #     id = int(parent_node.get("id")) 
    #     if abs(id - id_tmp) > 1:
    #         # print("id_before:", id)
    #         id = id_tmp + 1
    #         id_tmp = id
    #         # print("del_img_id:", del_img_id)
    #         # # print(type(del_img_id))
    #         # print("id+after:", id)
    #         # # print("id_", id_)
    #         parent_node.set("id", str(id))
    #         changeId_num = changeId_num + 1
    #     else:
    #         id_tmp = id

    lenght = len(root.getchildren())
    # print("lenght:", lenght)
    ###root:annotations下有两个无关和image并列的无关子节点，image的id从0开始，所以-3，倒叙排
    id_tmp = lenght - 3
    for parent_node in root.iter('image'): 
        # print(parent_node)
        id = int(parent_node.get("id")) 
        parent_node.set("id", str(id_tmp))
        id_tmp = id_tmp - 1
        changeId_num = changeId_num + 1
    
    print("共修改%s张图Id"%changeId_num)
    Topath = "/mnt/data/semantic-segmentation-pytorch/annotations/sum/"
    os.makedirs(Topath, exist_ok = True)
    tree.write(Topath + file_name + "_sortId.xml", encoding='utf-8')
    print("write xml to " + Topath + file_name + "_sortId.xml")

def delete_picture_node(del_xml_file_path):
    file = open(del_xml_file_path, 'rb')
    tree = ET.parse(file)
    root = tree.getroot()
    sum = len(root.getchildren())-2
    delImg_num = 0
    img_dir = "/mnt/data/task_out/0605/baidu_loss"
    delImgName_list = []
    for img in os.listdir(img_dir):
        print(img)
        img_prefix = img.split(".")[0][0:-1]
        print(img_prefix)
        if img_prefix not in delImgName_list:
            delImgName_list.append(img_prefix)
    # print(delImgName_list)
    print(len(delImgName_list))
    # delImgName_list = ["NOR_20220105_152234_M_00001_0475_","NOR_20220105_152234_M_00001_0605_", "NOR_20220105_152234_M_00001_0615_", "NOR_20220105_152234_M_00001_0710_", "NOR_20220105_154644_M_00001_1550_", "NOR_20220105_154644_M_00001_1554_", "NOR_20220323_185415_M_00001_0220_"]
    # id_list = [604,605,606,607,1136,1137,1138,1139,1132,1133,1134,1135]
    # for parent_node in root.iter('image'):
    # root.findall返回一个list   
    for parent_node in root.findall('image'):
        id = int(parent_node.get("id"))
        # print(id)
        # print(type(parent_node.get("id")))       
        img_name = os.path.split(parent_node.get("name"))[1]
        # print(img_name)
        img_name_split = img_name.split(".")[0][0:-1]
        if img_name_split in delImgName_list:
        # if id in id_list:
            print("img_name:", img_name)
            print("id:", id)
            root.remove(parent_node)
            delImg_num = delImg_num + 1
   
    print("共删除%s张图"%delImg_num)
    sum = sum - delImg_num
    print("还剩%s张图"%sum)
    Topath = "/mnt/data/semantic-segmentation-pytorch/annotations/sum/"
    os.makedirs(Topath, exist_ok = True)
    tree.write(Topath + "annotations_original_amout%s_0715.xml"%(sum), encoding='utf-8')
    print("write xml to " + Topath + "annotations_original_amout%s_0715.xml"%(sum))


def replace_node(xml_file_path, replace_xml_file_path):
    file = open(xml_file_path, 'rb')
    tree = ET.parse(file)
    root = tree.getroot()
    # print("root:", root)
    replace_file = open(replace_xml_file_path, 'rb')
    replace_tree = ET.parse(replace_file)
    replace_root = replace_tree.getroot()
    num = 0
    file_name = os.path.split(xml_file_path)[1].split(".")[0]
    print("file_name:",file_name)
    for replace_parent_node in replace_root.findall('image'):
        re_img_name = os.path.split(replace_parent_node.get("name"))[1]
        # print("re_img_name:", re_img_name)
        for parent_node in root.findall('image'):
            img_name = os.path.split(parent_node.get("name"))[1]
            if re_img_name == img_name:
                root.remove(parent_node)
                root.append(replace_parent_node)
                num += 1
            else:
                continue

    print("共替换%s张图"%num)
    Topath = "/mnt/data/semantic-segmentation-pytorch/annotations/sum/"
    os.makedirs(Topath, exist_ok = True)
    tree.write(Topath + file_name + "_replaced.xml", encoding='utf-8')
    print("write xml to " + Topath + file_name + "_replaced.xml")

def get_two_float(f_str, n):
    f_str = str(f_str)      # f_str = '{}'.format(f_str) 也可以转换为字符串
    a, b, c = f_str.partition('.')
    c = (c+"0"*n)[:n]       # 如论传入的函数有几位小数，在字符串后面都添加n为小数0
    return ".".join([a, c])

def add_json_node(xml_file_path, json_root_path):
    file = open(xml_file_path, 'rb')
    ###from lxml improt etree 才可用于新加入的json换行排列，否则会造成数据丢失，不能再用xml的ET
    parser = etree.XMLParser(remove_blank_text=True)
    tree =etree.parse(file, parser)
    root = tree.getroot()
    # print(tree)
    file_name = os.path.split(xml_file_path)[1].split(".")[0]
    # print("file_name:", file_name)
    dir_list = ["fs_src_0605_37du_json_rename", "fs_src_0605_baidu_json_rename"]
    id = 0
    for dir in dir_list:
        dir_path = os.path.join(json_root_path, dir)
        
        for file in os.listdir(dir_path):
            # print("file:", file)
            jsonFile_path = os.path.join(dir_path, file)
            f = open(jsonFile_path, 'r')
            content = f.read()
            img = json.loads(content)
            height = img['height']
            width = img['width']
            name_prefix = os.path.splitext(file)[0]
            name = name_prefix + ".jpg"
            # print("name:", name)
            image = etree.Element("image", height = str(height), width = str(width), id = str(id), name = str(name))
            # image.set('height', str(height))
            # image.set('width', str(width))
            # image.set('id', str(id))
            # image.set('name', str(name))
            if "road" in img.keys():
                if img["road"]:
                    r_points = img["road"][0]["points"]
                    ###把json中的list[x,y,x,y,x,y,...]转成cvat中xml支持的格式:"x,y;x,y;x,y;..."
                    # print(type(r_points))
                    # print(len(r_points))
                    road_points = list(np.array(r_points).reshape((int(len(r_points)/2)),2))
                    # print(road_points)
                    road_points_list = []
                    for r in road_points:
                        # print(type(i))
                        # print(i)
                        xy = str(r.tolist()).strip("[").strip("]")
                        # print(type(xy))
                        # print(xy)
                        x, y = get_two_float(xy.split(",")[0], 2), get_two_float(xy.split(",")[1], 2)
                        xy = x + "," + y
                        road_points_list.append(xy)
                    # print(len(road_points_list))
                    road_points_str =";".join(road_points_list)
                    road_points_str = road_points_str.replace(" ", "")
                    # print(road_points_str)
                    polygon = etree.SubElement(image, 'polygon')
                    polygon.set('label',"road")
                    polygon.set('occluded',"0")
                    polygon.set('points',road_points_str)
                    polygon.set('source',"manual")
                    polygon.set('z_order',"0")
                    image.append(polygon)
            if "cone_barrel" in img.keys():
                if img["cone_barrel"]:
                    for i in range(len(img["cone_barrel"])):
                        c_points = img["cone_barrel"][i][0]["points"]
                        # print(c_points)
                        # print(type(c_points))
                        # print(len(c_points))
                        cone_points = list(np.array(c_points).reshape((int(len(c_points)/2)),2))
                        # print(cone_points)
                        cone_points_list = []
                        for c in cone_points:
                            # print(type(c))
                            # print(c)
                            xy = str(c.tolist()).strip("[").strip("]")
                            # print(type(xy))
                            # print(xy)
                            x, y = get_two_float(xy.split(",")[0], 2), get_two_float(xy.split(",")[1], 2)
                            xy = x + "," + y
                            # print(type(xy))
                            # print(xy)
                            cone_points_list.append(xy)
                        # print(len(cone_points_list))
                        # print(cone_points_list)
                        cone_points_str =";".join(cone_points_list)
                        cone_points_str = cone_points_str.replace(" ", "")
                        # print(i,cone_points_str,"###########")
                        polygon = etree.SubElement(image, 'polygon')
                        polygon.set('label',"cone_barrel")
                        polygon.set('occluded',"0")
                        polygon.set('points',cone_points_str)
                        polygon.set('source',"manual")
                        polygon.set('z_order',"0")
                        image.append(polygon)
            if "ground_lock" in img.keys():
                if img["ground_lock"]:
                    for i in range(len(img["ground_lock"])):
                        g_points = img["ground_lock"][i][0]["points"]
                        ground_points = list(np.array(g_points).reshape((int(len(g_points)/2)),2))
                        ground_points_list = []
                        for g in ground_points:
                            xy = str(g.tolist()).strip("[").strip("]")
                            x, y = get_two_float(xy.split(",")[0], 2), get_two_float(xy.split(",")[1], 2)
                            xy = x + "," + y
                            ground_points_list.append(xy)
                        ground_points_str =";".join(ground_points_list)
                        ground_points_str = ground_points_str.replace(" ", "")
                        polygon = etree.SubElement(image, 'polygon')
                        polygon.set('label',"ground_lock")
                        polygon.set('occluded',"0")
                        polygon.set('points',ground_points_str)
                        polygon.set('source',"manual")
                        polygon.set('z_order',"0")
                        image.append(polygon)
            id = id + 1
            root.append(image)
    # tree = etree.ElementTree(root)
    # print(tree)
    
    Topath = "/mnt/data/semantic-segmentation-pytorch/annotations/sum/"
    os.makedirs(Topath, exist_ok = True)
    tree.write(Topath + file_name + "_addJson.xml",pretty_print=True, xml_declaration=False, encoding='utf-8')
    print("a total of %s images were modified!"%id)

def compare_delete_node(compare_del_xml_file_path):
    file = open(compare_del_xml_file_path, 'rb')
    tree = ET.parse(file)
    root = tree.getroot()
    sum = len(root.getchildren())-2
    delImg_num = 0
    img_path = "/mnt/data/images/"
    ImgName_list = []
    for root_, dirs_, files in os.walk(img_path):
        for file in files:
            img_prefix = file.split(".")[0]
            # print(img_prefix)
            if img_prefix not in ImgName_list:
                ImgName_list.append(img_prefix)
    print(len(ImgName_list))
    for parent_node in root.findall('image'):
        id = int(parent_node.get("id"))
        # print(id)
        # print(type(parent_node.get("id")))       
        img_name = os.path.split(parent_node.get("name"))[1]
        # print(img_name)
        img_name_split = img_name.split(".")[0]
        if img_name_split not in ImgName_list:
        # if id in id_list:
            print("img_name_split:", img_name_split)
            print("id:", id)
            root.remove(parent_node)
            delImg_num = delImg_num + 1
    
    print("共删除%s张图"%delImg_num)
    sum = sum - delImg_num
    print("还剩%s张图"%sum)
    Topath = "/mnt/data/semantic-segmentation-pytorch/annotations/sum/"
    os.makedirs(Topath, exist_ok = True)
    tree.write(Topath + "annotations_original_amout%s_0715_.xml"%(sum), encoding='utf-8')
    print("write xml to " + Topath + "annotations_original_amout%s_0715_.xml"%(sum))

def find_path(ImgName_dic_list, img_str):
    for ImgName_dic in ImgName_dic_list:
        if img_str in ImgName_dic.keys():
            print(ImgName_dic[img_str])
            img_str_path = ImgName_dic[img_str]
    return img_str_path

def compare_add_node(compare_add_xml_file_path):
    file_name = os.path.split(compare_add_xml_file_path)[1].split(".")[0]
    print("file_name:", file_name)
    file = open(compare_add_xml_file_path, 'rb')
    ###from lxml improt etree 才可用于新加入的json换行排列，否则会造成数据丢失，不能再用xml的ET
    parser = etree.XMLParser(remove_blank_text=True)
    tree =etree.parse(file, parser)
    root = tree.getroot()
    sum = len(root.getchildren())-2
    print("sum:", sum)
    addImg_num = 0
    img_path = "/mnt/data/images/"
    ImgName_list = []
    ImgName_dic_list = []
    for root_, dirs_, files in os.walk(img_path):
        for file in files:
            # print("root_:", root_)
            # print("file:", file)
            img_prefix = file.split(".")[0]
            # print(img_prefix)
            if img_prefix not in ImgName_list:
                ImgName_dic = {img_prefix:root_}
                ImgName_list.append(img_prefix)
                ImgName_dic_list.append(ImgName_dic)
    # print(ImgName_dic_list)
    print(len(ImgName_list))
    json_list = []
    for parent_node in root.findall('image'):
        id = int(parent_node.get("id"))
        # print(id)
        # print(type(parent_node.get("id")))       
        img_name = os.path.split(parent_node.get("name"))[1]
        img_name_split = img_name.split(".")[0]
        # print(img_name_split)
        if img_name_split not in json_list:
                json_list.append(img_name_split)
    print(len(json_list))
    retJson = list(set(ImgName_list).difference(set(json_list)))
    print(retJson)
    id = sum
    for img_str in retJson:
        print(img_str)
        img_str_path = find_path(ImgName_dic_list, img_str)
        img_name_ = img_str + ".jpg"
        img_path_ = os.path.join(img_str_path, img_name_)
        print(img_path_)
        img_name_array = cv2.imread(img_path_)
        # print(type(img_name_array))
        # print(img_name_array.shape)
        height, width = img_name_array.shape[0], img_name_array.shape[1] #高、宽
        image = etree.Element("image", height = str(height), width = str(width), id = str(id), name = str(img_path_))
        id = id + 1
        addImg_num = addImg_num + 1
        root.append(image)
    
    Topath = "/mnt/data/semantic-segmentation-pytorch/annotations/sum/"
    os.makedirs(Topath, exist_ok = True)
    tree.write(Topath + file_name + "_compare_add_node.xml",pretty_print=True, xml_declaration=False, encoding='utf-8')
    print("a total of %s images were modified!"%addImg_num)

if __name__ == '__main__':
    # sum_node(xml_file_path, add_xml_file_path)
    # delete_picture_node(del_xml_file_path)
    # sort_id(sort_xml_file_path)
    # replace_node(xml_file_path, replace_xml_file_path)
    # add_json_node(xml_file_path, json_root_path)
    # compare_delete_node(compare_del_xml_file_path)
    compare_add_node(compare_add_xml_file_path)
