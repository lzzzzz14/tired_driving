'''
本案例代码基于yolov7，实现人脸+手机识别。

注意：本案例必须使用GPU运行，请查看《ModelArts JupyterLab 硬件规格使用指南》了解切换硬件规格的方法
'''

# 拉取代码
import moxing as mox
mox.file.copy_parallel('obs://obs-aigallery-zc/clf/code/yolov7', 'yolov7')

INFO:root:Using MoXing-v2.1.0.5d9c87c8-5d9c87c8

INFO:root:Using OBS-Python-SDK-3.20.9.1

%cd yolov7

/home/ma-user/work/pilaojiashi/yolov7


# 配置环境
!pip install -r requirements.txt


# 查看数据集
import random
import cv2
import numpy as np
from  matplotlib import pyplot as plt
%matplotlib inline


with open("datasets/Fatigue_driving_detection/train.txt", "r") as f:
    img_paths = f.readlines()

img_paths = random.sample(img_paths, 8)
img_lists = []

for img_path in img_paths:
    img_path = img_path.strip()
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    tl = round(0.002 * (h + w) / 2) + 1
    color = [random.randint(0, 255) for _ in range(3)]
    if img_path.endswith('.png'):
        with open(img_path.replace("images", "txt_labels").replace(".png", ".txt")) as f:
            labels = f.readlines()
    if img_path.endswith('.jpeg'):
        with open(img_path.replace("images", "txt_labels").replace(".jpeg", ".txt")) as f:
            labels = f.readlines()
    for label in labels:
        l, x, y, wc, hc = [float(x) for x in label.strip().split()]
        
        cv2.rectangle(img, (int((x - wc / 2) * w), int((y - hc / 2) * h)), (int((x + wc / 2) * w), int((y + hc / 2) * h)), 
                      color, thickness=tl, lineType=cv2.LINE_AA)
    img_lists.append(cv2.resize(img, (1280, 720)))

image = np.concatenate([np.concatenate(img_lists[:4], axis=1), np.concatenate(img_lists[4:], axis=1)], axis=0)

plt.rcParams["figure.figsize"] = (20, 10)
plt.imshow(image[:,:,::-1])
plt.show()

# 在训练过程中使用的数据增强策略可以在使用的训练配置文件中设置，我们使用的是yolov7/data/hyp.scratch.tiny.yaml：
%pycat data/hyp.scratch.tiny.yaml

# 我们也可以查看数据增强后的数据：
import yaml
import argparse
import torch
from utils.datasets import create_dataloader
from utils.general import colorstr
from utils.plots import plot_images
%matplotlib inline

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default="data/coco.yaml", help='*.data path')
parser.add_argument('--hyp', type=str, default="data/hyp.scratch.tiny.yaml")
parser.add_argument('--batch-size', type=int, default=16, help='size of each image batch')
parser.add_argument('--gs', type=int, default=32)
parser.add_argument('--img-size', type=int, default=320, help='inference size (pixels)')
parser.add_argument('--task', default='train', help='train, val, test, speed or study')
parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
opt = parser.parse_args(args=['--task', 'train'])
print(opt)

with open(opt.data) as f:
    data = yaml.load(f, Loader=yaml.SafeLoader)
    
with open(opt.hyp) as f:
    hyp = yaml.load(f, Loader=yaml.SafeLoader)
        
dataloader = create_dataloader(data[opt.task], opt.img_size, opt.batch_size, opt.gs, opt, hyp=hyp, augment=True, pad=0.5, rect=True, prefix=colorstr(f'{opt.task}: '))[0]

for img, targets, paths, shapes in dataloader:
    nb, _, height, width = img.shape
    targets[:, 2:] *= torch.Tensor([width, height, width, height])
    result = plot_images(img, targets)
    break
    
plt.rcParams["figure.figsize"] = (20, 10)
plt.imshow(result)
plt.show()


# 训练模型

# yolov7是anchor base的方法，utils/autoanchor中提供了计算anchor的函数：

from utils.autoanchor import kmean_anchors
kmean_anchors(path='data/coco.yaml', n=9, img_size=320, thr=4.0, gen=1000, verbose=False)

# 查看模型配置文件
%pycat cfg/lite/yolov7-tiny.yaml

from models.yolo import Model
import thop

# 我们可以构建模型，查看模型flops与params参数：
model = Model("cfg/lite/yolov7-tiny.yaml", ch=3, nc=2, anchors=hyp.get('anchors'))
inputs = torch.randn(1, 3, 288, 352)
flops, params = thop.profile(model, inputs=(inputs,))
print('flops:', flops / 900000000 * 2)
print('params:', params)

# 微调模型
# 提供的预训练模型已经基于数据训练过248个epochs，我们在预训练模型上训练5个epochs：
!python train.py --model_name yolov7-tiny --batch-size 32 --epochs 5 --name c-320 --multi-scale --sync-bn --device 0

# 训练结束，可以看到outputs文件夹下保存了训练结果，加载best.pt查看推理结果：
!python detect.py --weights outputs/c-32029/weights/best.pt --source images --img-size 640 --device 0 --name c-32029 --no-trace

# 可以看到结果保存在runs/detect/c-320，查看结果：
import cv2
from  matplotlib import pyplot as plt
%matplotlib inline

img = cv2.imread("runs/detect/c-32029/day_man_001_30_1_176.png")
plt.imshow(img[:,:,::-1])
plt.show()

