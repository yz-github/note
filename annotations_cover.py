from traceback import print_tb
import xml.etree.ElementTree as ET
import os
import json

def get_key(dict, value):
    return[k for k, v in dict.items() if v == value]

def find_json_child(jsonFile_path):
    f = open(jsonFile_path, 'r')
    content = f.read()
    a = json.loads(content)
    # print(type(a))
    value = a["polygon"]
    points = []
    for v in value:
        # print(type(v))
        # print(v)
        list_new = [str(xy) for xy in v]
        # print(list_new)
        point =",".join(list_new)
        # print("point:", point)
        # print(type(point))
        points.append(point)
    # print("points:",points)
    
    points = ";".join([x for x in points])
    f.close()
    return points

def find_BFLR_child(root):
    for parent_node in root.iter('image'):
        img_name = parent_node.get("name")
        id = int(parent_node.get("id"))
        filename = os.path.split(img_name)[1]
        img_suffix = filename.split("_")[-1]           
        if id == 0:
            for B_child in parent_node:
                b_polygon_child = B_child
                # print("B_child:", B_child)
                # print("parent_node:", parent_node)
                # print("img_name:", img_name)
                # print("id:", id)
                # print("filename:", filename)
                # print("img_suffix:", img_suffix)
        if id == 1:
            for F_child in parent_node:
                f_polygon_child = F_child
        if id == 2:
            for L_child in parent_node:
                l_polygon_child = L_child
                # print("L_child:", L_child)
                # print("parent_node:", parent_node)
                # print("img_name:", img_name)
                # print("id:", id)
                # print("filename:", filename)
                # print("img_suffix:", img_suffix)       
        if id == 3:
            for R_child in parent_node:
                r_polygon_child = R_child
    return b_polygon_child, f_polygon_child, l_polygon_child, r_polygon_child

def ori_cover(xml_path, new_xml_path):
    file = open(xml_path, 'rb')
    tree = ET.parse(file)
    root = tree.getroot()
    #待写入xml
    file = open(new_xml_path, 'rb')
    new_tree = ET.parse(file)
    new_root = new_tree.getroot()
    num = 0
    for parent_node in new_root.iter('image'):      
        img_name = parent_node.get("name")
        id = int(parent_node.get("id"))
        filename = os.path.split(img_name)[1]
        img_suffix = filename.split("_")[-1]
        b_child, f_child, l_child, r_child = find_BFLR_child(root)
        # if id not in id_list:
            # 获取子节点
            # achild = parent_node.getchildren()
            # print(achild)
        if str(img_suffix) == "B.jpg":             
            parent_node.append(b_child)
            # child = parent_node.getchildren()
            # print(child)      
            num = num + 1 
        if str(img_suffix) == "F.jpg":              
            parent_node.append(f_child)
            num = num + 1 
        if str(img_suffix) == "L.jpg":              
            parent_node.append(l_child)
            num = num + 1            
        if str(img_suffix) == "R.jpg":              
            parent_node.append(r_child)
            num = num + 1 
        else:
            continue
    print("a total of %s images were modified!"%num)
    Topath = "/annotations/cover/"
    os.makedirs(Topath, exist_ok = True)
    new_tree.write(Topath + "annotations_cover.xml", encoding='utf-8')

def bird_cover(xml_path, jsonFile_dir):
    file = open(xml_path, 'rb')
    tree = ET.parse(file)
    root = tree.getroot()
    num = 0
    file_name = os.path.split(xml_path)[1].split(".")[0]
    print("file_name:", file_name)
    for parent_node in root.iter('image'):      
        img_name = parent_node.get("name")
        filename = os.path.split(img_name)[1]
        img_suffix = filename.split(".")[0]
        # print("img_suffix:", img_suffix)
        json_name = img_suffix +".json"
        for jsonFile in os.listdir(jsonFile_dir):
            if jsonFile.endswith("json") and jsonFile == json_name:
                # print("parent_node:", parent_node)
                # print("parent_node.tag:", parent_node.tag)
                # print("parent_node.attrib:", parent_node.attrib)
                print("filename:", filename)
                print(jsonFile)
                jsonFile_path = jsonFile_dir + jsonFile
                print("jsonFile_path:", jsonFile_path)
                circle_points = find_json_child(jsonFile_path)
                # print("circle_points:", circle_points)
                label = "road"
                polygon = ET.SubElement(parent_node, 'polygon', {'label':label, 'occluded':'0', 'points':circle_points, 'source':'manual', 'z_order':'0'})
                num += 1
    print("a total of %s images were modified!"%num)
    Topath = "/mnt/data/semantic-segmentation-pytorch/annotations/cover/"
    os.makedirs(Topath, exist_ok = True)
    tree.write(Topath + file_name + "_cover.xml", encoding='utf-8')

if __name__ == "__main__":
    xml_path = "/annotations_yadi.xml"
    new_xml_path = "/annotations.xml"
    jsonFile_dir = "/valScene/"
    
    ###原图替换
    # ori_cover(xml_path, new_xml_path)
    ###俯视图替换
    bird_cover(xml_path, jsonFile_dir)
