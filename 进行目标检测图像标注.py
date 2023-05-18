'''
我使用的是ModelArts标注
2.1 将上面的关键帧图像拷贝到你自己的OBS桶文件目录A中
2.2 点击此处进入ModelArts数据管理页面，新建物体检测数据集，数据输入选择上述的OBS桶文件目录A，输出任意选择OBS桶内其他目录B
2.3 开始标注，标注人脸标签face和手机标签phone（我标注的时候包括车内所有人脸），标注一部分（如200张图像）即可
2.4 开始智能标注，智能标注需要等待几十分钟，将智能标注后的图像进行核对确认
2.5 导出所有标注完的图像到OBS桶文件夹C中（尽量和A、B区分开）
2.6 将OBS桶文件夹C中图像拷贝到ModelArts Notebook进行转换（因为训练所需为TXT标注格式、标注转换代码如下）
2.7 划分训练集、验证集、测试集（划分代码如下）
'''

'***转换xml标注文件为txt格式，无法直接运行***'
import copy
from lxml.etree import Element, SubElement, tostring, ElementTree

import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

classes = ["face", "phone"]  # 类别


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(image_id):
    in_file = open('./label_xml/%s.xml' % (image_id), encoding='UTF-8')

    out_file = open('yolov7/datasets/Fatigue_driving_detection/txt_labels/%s.txt' % (image_id), 'w')  # 生成txt格式文件, 保存在yolov7训练所需的数据集路径中
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        cls = obj.find('name').text
        print(cls)
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

xml_path = './label_xml/'  # xml_path应该是上述步骤OBS桶文件夹C中的所有文件，记得拷贝过来

img_xmls = os.listdir(xml_path)
for img_xml in img_xmls:
    label_name = img_xml.split('.')[0]
    if  img_xml.split('.')[1] == 'xml':
        convert_annotation(label_name)

'***转换xml标注文件为txt格式，无法直接运行***'        
import moxing as mox
from random import sample
file_list = mox.file.list_directory( xml_path)  # xml_path中是上述步骤OBS桶文件夹C中的所有文件，记得拷贝到本地
print(len(file_list))
val_file_list = sample(file_list, 300)  # 选择了300张做测试集
line = ''
for i in val_file_list:
    if i.endswith('.png') :
        line += 'datasets/Fatigue_driving_detection/images/'+i+'\n'     # datasets/Fatigue_driving_detection/images/ 是yolov7训练使用的
with open('yolov7/datasets/Fatigue_driving_detection/val.txt', 'w+') as f:  
    f.writelines(line)

test_file_list = sample(file_list, 300)
line = ''
for i in test_file_list:
    if i.endswith('.png'):
        line += 'datasets/Fatigue_driving_detection/images/'+i+'\n'
with open('yolov7/datasets/Fatigue_driving_detection/test.txt', 'w+') as f:
    f.writelines(line)

line = ''
for i in file_list:
    if i not in val_file_list and i not in test_file_list:
        if i.endswith('.png') :
            line += 'datasets/Fatigue_driving_detection/images/'+i+'\n'
with open('yolov7/datasets/Fatigue_driving_detection/train.txt', 'w+') as f:
    f.writelines(line)