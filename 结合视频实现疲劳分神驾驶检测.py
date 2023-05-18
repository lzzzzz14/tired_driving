# 如果你第3步训练得到了更好的目标检测模型，将best.pt拷贝替换即可

# 拷贝提供的baseline 模型包到本地
import moxing as mox
mox.file.copy_parallel('obs://obs-aigallery-zc/clf/model/video_classification','video_classification')

# 推理代码 展示
from PIL import Image
import copy
import sys
import traceback
import os
import numpy as np
import time
import cv2
from input_reader import InputReader
from tracker import Tracker
from EAR import eye_aspect_ratio
from MAR import mouth_aspect_ratio

from models.experimental import attempt_load
from utils1.general import check_img_size
from tempfile import NamedTemporaryFile
from utils1.torch_utils import TracedModel
from detect import detect
from model_service.pytorch_model_service import PTServingBaseService


class fatigue_driving_detection(PTServingBaseService):
    def __init__(self, model_name, model_path):
        # these three parameters are no need to modify
        self.model_name = model_name
        self.model_path = model_path

        self.capture = 'test.mp4'

        self.width = 1920
        self.height = 1080
        self.fps = 30
        self.first = True

        self.standard_pose = [180, 40, 80]
        self.look_around_frame = 0
        self.eyes_closed_frame = 0
        self.mouth_open_frame = 0
        self.use_phone_frame = 0
        # lStart, lEnd) = (42, 48)
        self.lStart = 42
        self.lEnd = 48
        # (rStart, rEnd) = (36, 42)
        self.rStart = 36
        self.rEnd = 42
        # (mStart, mEnd) = (49, 66)
        self.mStart = 49
        self.mEnd = 66
        self.EYE_AR_THRESH = 0.2
        self.MOUTH_AR_THRESH = 0.6
        self.frame_3s = self.fps * 3
        self.face_detect = 0

        self.weights = "best.pt"
        self.imgsz = 640

        self.device = 'cpu'  # 大赛后台使用CPU判分

        model = attempt_load(model_path, map_location=self.device)
        self.stride = int(model.stride.max())
        self.imgsz = check_img_size(self.imgsz, s=self.stride)

        self.model = TracedModel(model, self.device, self.imgsz)


        self.need_reinit = 0
        self.failures = 0

        self.tracker = Tracker(self.width, self.height, threshold=None, max_threads=4, max_faces=4,
                          discard_after=10, scan_every=3, silent=True, model_type=3,
                          model_dir=None, no_gaze=False, detection_threshold=0.6,
                          use_retinaface=0, max_feature_updates=900,
                          static_model=True, try_hard=False)

        # self.temp = NamedTemporaryFile(delete=False)  # 用来存储视频的临时文件

    def _preprocess(self, data):
        # preprocessed_data = {}
        for k, v in data.items():
            for file_name, file_content in v.items():
                try:
                    try:
                        with open(self.capture, 'wb') as f:
                            file_content_bytes = file_content.read()
                            f.write(file_content_bytes)

                    except Exception:
                        return {"message": "There was an error loading the file"}

                    # self.capture = self.temp.name  # Pass temp.name to VideoCapture()
                except Exception:
                    return {"message": "There was an error processing the file"}
        return 'ok'

    def _inference(self, data):
        """
        model inference function
        Here are a inference example of resnet, if you use another model, please modify this function
        """
        print(data)
        result = {"result": {"category": 0, "duration": 6000}}

        self.input_reader = InputReader(self.capture, 0, self.width, self.height, self.fps)
        source_name = self.input_reader.name
        now = time.time()
        while self.input_reader.is_open():
            if not self.input_reader.is_open() or self.need_reinit == 1:
                self.input_reader = InputReader(self.capture, 0, self.width, self.height, self.fps, use_dshowcapture=False, dcap=None)
                if self.input_reader.name != source_name:
                    print(f"Failed to reinitialize camera and got {self.input_reader.name} instead of {source_name}.")
                    # sys.exit(1)
                self.need_reinit = 2
                time.sleep(0.02)
                continue
            if not self.input_reader.is_ready():
                time.sleep(0.02)
                continue

            ret, frame = self.input_reader.read()

            self.need_reinit = 0

            try:
                if frame is not None:
                    # 剪裁主驾驶位
                    frame = frame[:, 600:1920, :]

                    # 检测驾驶员是否接打电话 以及低头的人脸
                    bbox = detect(self.model, frame, self.stride, self.imgsz)
                    # print(results)

                    for box in bbox:
                        if box[0] == 0:
                            self.face_detect = 1
                        if box[0] == 1:
                            self.use_phone_frame += 1

                    # 检测驾驶员是否张嘴、闭眼、转头
                    faces = self.tracker.predict(frame)
                    if len(faces) > 0:

                        face_num = 0
                        max_x = 0
                        for face_num_index, f in enumerate(faces):
                            if max_x <= f.bbox[3]:
                                face_num = face_num_index
                                max_x = f.bbox[3]

                        f = faces[face_num]
                        f = copy.copy(f)
                        # 检测是否转头
                        if np.abs(self.standard_pose[0] - f.euler[0]) >= 45 or np.abs(self.standard_pose[1] - f.euler[1]) >= 45 or \
                                np.abs(self.standard_pose[2] - f.euler[2]) >= 45:
                            self.look_around_frame += 1
                        else:
                            self.look_around_frame = 0

                        # 检测是否闭眼
                        leftEye = f.lms[self.lStart:self.lEnd]
                        rightEye = f.lms[self.rStart:self.rEnd]
                        leftEAR = eye_aspect_ratio(leftEye)
                        rightEAR = eye_aspect_ratio(rightEye)
                        # average the eye aspect ratio together for both eyes
                        ear = (leftEAR + rightEAR) / 2.0

                        if ear < self.EYE_AR_THRESH:
                            self.eyes_closed_frame += 1
                        else:
                            self.eyes_closed_frame = 0
                        # print(ear, eyes_closed_frame)

                        # 检测是否张嘴
                        mar = mouth_aspect_ratio(f.lms)

                        if mar > self.MOUTH_AR_THRESH:
                            self.mouth_open_frame += 1
                    else:
                        if self.face_detect:
                            self.look_around_frame += 1
                            self.face_detect = 0
                    # print(self.look_around_frame)
                    if self.use_phone_frame >= self.frame_3s:
                        result['result']['category'] = 3
                        break

                    elif self.look_around_frame >= self.frame_3s:
                        result['result']['category'] = 4
                        break

                    elif self.mouth_open_frame >= self.frame_3s:
                        result['result']['category'] = 2
                        break

                    elif self.eyes_closed_frame >= self.frame_3s:
                        result['result']['category'] = 1
                        break
                    else:
                        result['result']['category'] = 0

                    self.failures = 0
                else:
                    break
            except Exception as e:
                if e.__class__ == KeyboardInterrupt:
                    print("Quitting")
                    break
                traceback.print_exc()
                self.failures += 1
                if self.failures > 30:   # 失败超过30次就默认返回
                    break
            del frame
        final_time = time.time()
        duration = int(np.round((final_time - now) * 1000))
        result['result']['duration'] = duration
        return result

    def _postprocess(self, data):
        # os.remove(self.temp.name)
        return data

'''
将模型导入到ModelArts,这里使用了ModelArts的SDK，你也可以在ModelArts控制台界面直接导入模型
完成这一步，可以点击此处，前往AI应用管理界面提交模型，如果还需要进行测试，可以部署在线服务进行测试

导入需要几分钟时间，请耐心等待
'''

from modelarts.session import Session
from modelarts.model import Model
from modelarts.config.model_config import TransformerConfig,Params
!pip install json5
import json5
import re
import traceback
import random

try:
    session = Session()
    config_path = 'video_classification/model/config.json' 
    if mox.file.exists(config_path):                                        # 判断一下是否存在配置文件，如果没有则不能导入模型
        model_location =  'video_classification/model'
        model_name = "video_classification"
        load_dict = json5.loads(mox.file.read(config_path))
        model_type = load_dict['model_type']
        re_name = '_'+str(random.randint(0,1000))
        model_name += re_name
        print("正在导入模型,模型名称：", model_name)
        model_instance = Model(
                     session, 
                     model_name=model_name,               # 模型名称
                     model_version="1.0.0",               # 模型版本
                      source_location_type='LOCAL_SOURCE',
                     source_location=model_location,      # 模型文件路径
                     model_type=model_type,               # 模型类型
                     )

    print("所有模型导入完成")
except Exception as e:
    print("发生了一些问题，请看下面的报错信息：") 
    traceback.print_exc()
    print("模型导入失败")
