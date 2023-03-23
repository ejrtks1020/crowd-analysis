# -*- coding: utf-8 -*-
# @Organization  : insightface.ai
# @Author        : Jia Guo
# @Time          : 2021-05-04
# @Function      : 


from __future__ import division

import glob
import os.path as osp
import time

import numpy as np
import onnxruntime
import cv2


from numpy.linalg import norm

from ..model_zoo import model_zoo
from ..utils import DEFAULT_MP_NAME, ensure_available, face_align
from .common import Face

__all__ = ['FaceAnalysis', 'face_align']
MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)

class FaceAnalysis:
    def __init__(self, name=DEFAULT_MP_NAME, root='~/.insightface', allowed_modules=None, gender='caffe',
                            min_box_size=0, **kwargs):
        onnxruntime.set_default_logger_severity(3)
        self.models = {}
        self.model_dir = ensure_available('models', name, root=root)
        onnx_files = glob.glob(osp.join(self.model_dir, '*.onnx'))
        onnx_files = sorted(onnx_files)
        for onnx_file in onnx_files:
            model = model_zoo.get_model(onnx_file, gender, **kwargs)
            if model is None:
                print('model not recognized:', onnx_file)
            elif allowed_modules is not None and model.taskname not in allowed_modules:
                print('model ignore:', onnx_file, model.taskname)
                del model
            elif model.taskname not in self.models and (allowed_modules is None or model.taskname in allowed_modules):
                print('find model:', onnx_file, model.taskname, model.input_shape, model.input_mean, model.input_std)
                self.models[model.taskname] = model
            else:
                print('duplicated model task type, ignore:', onnx_file, model.taskname)
                del model
        assert 'detection' in self.models
        self.det_model = self.models['detection']

        self.min_box_size = min_box_size
        self.gender = gender
        if gender == 'youtube':                            
            self.gender_model = onnxruntime.InferenceSession('gender_full_data.onnx')
            self.outname = [i.name for i in self.gender_model.get_outputs()]
            self.inname = [i.name for i in self.gender_model.get_inputs()]
        elif gender == 'aihub':                            
            self.gender_model = onnxruntime.InferenceSession('/home/xinapse/ailab-insightface-modified/aihub_aug_gray_4.onnx')
            self.outname = [i.name for i in self.gender_model.get_outputs()]
            self.inname = [i.name for i in self.gender_model.get_inputs()]
          
        self.num = 0

    def prepare(self, ctx_id, det_thresh=0.5, det_size=(640, 640)):
        self.det_thresh = det_thresh
        assert det_size is not None
        print('set det-size:', det_size)
        self.det_size = det_size
        for taskname, model in self.models.items():
            if taskname=='detection':
                model.prepare(ctx_id, input_size=det_size, det_thresh=det_thresh)
            else:
                model.prepare(ctx_id)

    def get(self, img, max_num=0):
        total_speed = 0
        st = time.time()
        bboxes, kpss = self.det_model.detect(img,
                                             max_num=max_num,
                                             metric='default')
        ed = time.time()
        # print(f"detection speed : {round((ed - st) * 1000,2)}")
        total_speed += round((ed - st) * 1000,2)

        if bboxes.shape[0] == 0:
            return []
        ret = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            for taskname, model in self.models.items():
                if taskname=='detection':
                    continue
                st = time.time()
                model.get(img, face)
                ed = time.time()
                total_speed += round((ed - st) * 1000,2)
                # print(f"{taskname} speed : {round((ed - st) * 1000,2)}ms")
            ret.append(face)
        
        # print(f"Total Speed : {total_speed}ms")
        return ret

    def new_get(self, img, max_num=0):
        total_speed = 0
        st = time.time()
        bboxes, kpss = self.det_model.detect(img,
                                             max_num=max_num,
                                             metric='default')
        ed = time.time()
        # print(f"detection speed : {round((ed - st) * 1000,2)}")
        total_speed += round((ed - st) * 1000,2)

        if self.min_box_size:
            over_min_box = [[bbox, kps] for bbox, kps in zip(bboxes, kpss) if (bbox[2] - bbox[0]) > self.min_box_size and (bbox[3] - bbox[1]) > self.min_box_size]
            bboxes = np.asarray([bbox_kps[0] for bbox_kps in over_min_box])
            kpss = np.asarray([bbox_kps[1] for bbox_kps in over_min_box])

        # for box in bboxes:
        #     print(f"Width : {box[2] - box[0]} Height : {box[3] - box[1]}")
        # kpss = np.array([kps for kps in zip(bboxes, kpss) if (bbox[2] - bbox[0]) > 96 and (bbox[3] - bbox[1]) > 96])

        if bboxes.shape[0] == 0:
            return []

        faces = []
        for bbox in bboxes.astype(int):
            w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
            center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
            rotate = 0
            _scale = 96  / (max(w, h)*1.5)
            aimg, M = face_align.transform(img, center, 96, _scale, rotate)
            # aimg = self.change_brightness(aimg, 20)
            if self.gender == 'aihub':
                aimg = (cv2.cvtColor(aimg.astype(np.float32), cv2.COLOR_RGB2GRAY) / 255.)[..., None]
            # aimg = np.transpose(aimg, (2,0,1))
            faces.append(aimg)
        # bbox_faces = np.array([cv2.resize(img[max(0,bbox[1]):max(0, bbox[3]), max(0,bbox[0]):], (96, 96))[:,:,::-1] for bbox in bboxes.astype(np.int32)], dtype=np.float32)
        # st = time.time()
        if self.gender == 'aihub':
            genders = self.gender_model.run(self.outname, {self.inname[0] : faces})    
        else:
            genders = self.gender_model.run(self.outname, {self.inname[0] : faces})[0]
        # print(time.time() - st)

        ##############################################################################
        overlapped_bboxes = {}
        non_overlapped_bboxes = []
        new_kpss = []
    
        ret = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            for taskname, model in self.models.items():
                # if taskname=='detection' or taskname=='recognition':
                if taskname=='detection':
                    continue

                if taskname=='recognition' and i in overlapped_bboxes.keys():
                    face.embedding = overlapped_bboxes[i][0]
                    if self.gender == 'arcface':
                        face.gender = overlapped_bboxes[i][1]
                    continue

                st = time.time()
                model.get(img, face)
                ed = time.time()
                # print(f"{taskname} speed :{round((ed - st) * 1000,2)}ms")
                total_speed += round((ed - st) * 1000,2)
            # face.gender = genderPreds[i]
            if self.gender != 'default' and self.gender != "arcface":
                # man_condition = gender_prob[0] > gender_prob[1] if gender_model == 'caffe' else gender_prob[0] < gender_prob[1]

                if self.gender == 'aihub':
                    # print(np.argmax(genders[0][i]))
                    # face['age'] = int(genders[0][i] * 86)
                    # face['gender'] = genders[1][i]


                    ############# aug and new age cls
                    face['age'] = np.argmax(genders[0][i]) * 10
                    face['gender'] = genders[1][i]
                else:
                    face.gender = genders[i]

            ret.append(face)
        return ret

    def change_brightness(self, img, value=30):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v,value)
        v[v > 255] = 255
        v[v < 0] = 0
        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
        return img

    def check_face_front(self, kps):
        nose_x = kps[2][0]
        nose_y = kps[2][1]
        right_eye_x = kps[0][0]
        right_eye_y = kps[0][1]
        left_eye_x = kps[1][0]
        left_eye_y = kps[1][1]
        right_mouse_x = kps[3][0]
        right_mouse_y = kps[3][1]
        left_mouse_x = kps[4][0]
        left_mouse_y = kps[4][1]
        eye_center_x = (left_eye_x + right_eye_x) // 2
        eye_center_y = (left_eye_y + right_eye_y) // 2
        mouse_center_x = (left_mouse_x + right_mouse_x) // 2
        mouse_center_y = (left_mouse_y + right_mouse_y) // 2
        check_nose_x = nose_x > right_eye_x and nose_x < left_eye_x
        check_nose_y = nose_y < mouse_center_y and nose_y > eye_center_y
        if nose_x <= right_eye_x or nose_x <= right_mouse_x:
            return 'right'
        if nose_x >= left_eye_x or nose_x >= left_mouse_x:
            return 'left'
        else:
            return 'front'     

    def iou(self, box1, box2):
        # box = (x1, y1, x2, y2)
        box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

        # obtain x1, y1, x2, y2 of the intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # compute the width and height of the intersection
        w = max(0, x2 - x1 + 1)
        h = max(0, y2 - y1 + 1)

        inter = w * h
        iou = inter / (box1_area + box2_area - inter)
        return iou

    def draw_on(self, img, faces):
        import cv2
        dimg = img.copy()
        for i in range(len(faces)):
            face = faces[i]
            box = face.bbox.astype(np.int)
            color = (0, 0, 255)
            cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 2)
            if face.kps is not None:
                kps = face.kps.astype(np.int)
                #print(landmark.shape)
                for l in range(kps.shape[0]):
                    color = (0, 0, 255)
                    if l == 0 or l == 3:
                        color = (0, 255, 0)
                    cv2.circle(dimg, (kps[l][0], kps[l][1]), 1, color,
                               2)
            if face.gender is not None and face.age is not None:
                cv2.putText(dimg,'%s,%d'%(face.sex,face.age), (box[0]-1, box[1]-4),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),1)

            #for key, value in face.items():
            #    if key.startswith('landmark_3d'):
            #        print(key, value.shape)
            #        print(value[0:10,:])
            #        lmk = np.round(value).astype(np.int)
            #        for l in range(lmk.shape[0]):
            #            color = (255, 0, 0)
            #            cv2.circle(dimg, (lmk[l][0], lmk[l][1]), 1, color,
            #                       2)
        return dimg

