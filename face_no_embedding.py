import enum
import cv2
import time
import requests
import random
import numpy as np
import onnxruntime as ort
from PIL import Image
from pathlib import Path
from collections import OrderedDict,namedtuple
import fire
import timeit
import time
import glob
from insightface.app import FaceAnalysis
import tqdm
import logging
import os
from utils3 import check_face_front, iou, cosine_similarity, letterbox
# from annoy import AnnoyIndex
import torch
import torch.nn as nn
import sqlite3
# import psutil



NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 
         'hair drier', 'toothbrush']

COLORS = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(NAMES)}         
CUDA = False


def preprocess(img):
    img_head = img.copy()
    img_head, ratio, dwdh = letterbox(img_head, auto=False)
    img_head = img_head.transpose((2, 0, 1))
    img_head = np.expand_dims(img_head, 0)
    img_head = np.ascontiguousarray(img_head)

    im = img_head.astype(np.float32)
    im /= 255
    return ratio,dwdh,im  

def video(path,
          coreml,
          face_model,
          face_det_size,
          draw,
          draw_age_gender,
          filename,
          save_video,
          save_log,
          time_out,
          disappear_tolerance,
          use_prev_results,
          iou_threshold,
          save_original,
          min_appear,
          gender_model,
          blur_face,
          min_box_size,
          head_detector):
    providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider'] if coreml else ['CPUExecutionProvider']
    if head_detector:
        onnx_weight = 'yolo-tiny-head.onnx'
        session = ort.InferenceSession(onnx_weight, providers=providers)
        outname = [i.name for i in session.get_outputs()]
        inname = [i.name for i in session.get_inputs()]
    if gender_model == 'aihub':
        app = FaceAnalysis(name=face_model, allowed_modules=['detection'], 
                           providers=providers, gender=gender_model, root='.',
                           min_box_size=min_box_size)
    else:
        app = FaceAnalysis(name=face_model, allowed_modules=['detection', 'genderage'],
                           providers=providers, gender=gender_model, root='.',
                           min_box_size=min_box_size) 
        # app = FaceAnalysis(name=face_model, allowed_modules=['detection', 'recognition'],
        #                    providers=providers, gender=gender_model) # enable detection model only                           
    app.prepare(ctx_id=0, det_size=(face_det_size, face_det_size))
    currentDT = time.localtime()
    start_datatime = time.strftime("%y-%m-%d-%H-%M-%S", currentDT)
    filename = filename

    if not os.path.exists('insightface_output'):
        os.makedirs('insightface_output')
    if not os.path.exists('insightface_output/' + filename):
        os.makedirs('insightface_output/' + filename)

    if path == 'webcam':
        cam = cv2.VideoCapture(0)
        mode = 'webcam'
        # video_fps = int(cam.get(cv2.CAP_PROP_FPS))
        video_fps = 10
        cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cam.set(cv2.CAP_PROP_FPS, 30)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)        
        video_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # print(f"Webcam FPS : {video_fps}")
        print("Webcam FPS : ", cam.get(cv2.CAP_PROP_FPS))
        print(f"Webcam width : {video_width}")
        print(f"Webcam height : {video_height}")

    elif '*' in path:
        files = sorted(glob.glob(path))
        length = len(files)
        mode = 'dir'

    elif os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, '*.*')))
        length = len(files)
        mode = 'dir'

    elif path.endswith(('mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv')):
        cam = cv2.VideoCapture(path)
        video_n_frames = cam.get(cv2.CAP_PROP_FRAME_COUNT)
        video_fps = cam.get(cv2.CAP_PROP_FPS)
        video_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        length = video_n_frames
        mode = 'video'

    assert mode == 'video' or mode == 'dir' or mode == 'webcam'

    if mode == 'dir':
        im0 = cv2.imread(files[0])
        video_height, video_width = im0.shape[0], im0.shape[1]
        video_fps = 15

    video_path = f'insightface_output/{filename}/{filename}-{start_datatime}.mp4'
    if save_video:
        out_video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(
            *'mp4v'), video_fps, (video_width, video_height))
        if mode == 'webcam' and save_original:
            original_video_path = f'insightface_output/{filename}/original-{filename}-{start_datatime}.mp4'
            out_original_video = cv2.VideoWriter(original_video_path, cv2.VideoWriter_fourcc(
            *'mp4v'), 15, (video_width, video_height))



    log_file = f'insightface_output/{filename}/{filename}-{start_datatime}.txt'
    if os.path.exists(log_file):
        os.remove(log_file)
    if save_log:
        logging.basicConfig(filename=log_file, level=logging.INFO, format='%(message)s')


    
    if save_log or save_video:   
        conn = sqlite3.connect(f'insightface_output/{filename}/{filename}-{start_datatime}.db')
        cursor = conn.cursor()
        cursor.execute(f"CREATE TABLE IF NOT EXISTS pedestrians \
                        (id, face_id, gender, age, watching_time, stored_time, appear_count)")

    total_fps = 0
    min_fps = 999
    max_fps = 0

    total_response_time = 0
    min_response_time = 999999
    max_response_time = 0
    cnt = 0
    face_info_dict = {}
    informations = {
        # "total_pedestrian_num" : 0,
        # "gender" : {"M" : 0, "F" : 0},
        # "age" : {i : 0 for i in range(0, 110, 10)}, # 0 ~ 100살까지
        "look_count" : {"M" : {i : 0 for i in range(0, 110, 10)},
                        "F" : {i : 0 for i in range(0, 110, 10)}},        
        "person_count" : {"M" : {i : 0 for i in range(0, 110, 10)},
                        "F" : {i : 0 for i in range(0, 110, 10)}}
    }
    face_id = 0

    prev_n_face_info_dict = {}
    prev_face_info_dict = {}
    prev_result = []

    people_count = 0
    prev_heads = []
    head_color = [random.randint(0, 255) for _ in range(3)]
    face_time_out = {}
    if mode != 'webcam':
        pbar = tqdm.tqdm(total=length, position=0, leave=True)
    while True:
        if mode == 'video' or mode =='webcam':
            ret, img = cam.read()
            if not ret:
                break
        
        if mode == 'dir':
            try:
                img = cv2.imread(files[cnt])
            except:
                break
        
        temp_result = []
        cnt += 1
        if save_video and mode == 'webcam' and save_original:
            out_original_video.write(img)
        fps_start = timeit.default_timer()
        st_total = time.time()

        # for draw
        image = img.copy()
        
        # for head model
        if head_detector:
            ratio, dwdh, im = preprocess(img)
            outputs = session.run(outname, {inname[0] : im})[0]
            overlapped_prev = []
            new_heads = []
            for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(outputs):
                if cls_id == 0:
                    box = np.array([x0,y0,x1,y1])
                    box -= np.array(dwdh*2)
                    box /= ratio
                    box = box.round().astype(np.int32).tolist()
                    if prev_heads:
                        overlapped = False
                        max_iou = 0
                        overlapped_index = 0
                        for i, prev_box in enumerate(prev_heads):
                            if iou(box, prev_box[0]) > 0.2:
                                if max_iou < iou(box, prev_box[0]):
                                    max_iou = iou(box, prev_box[0])
                                    overlapped_index = i
                                overlapped = True
                        
                        if overlapped:
                            prev_heads[overlapped_index][0] = box
                            overlapped_prev.append(overlapped_index)
                        else:
                            new_heads.append([box, 0])
                            people_count += 1

                    else:
                        new_heads.append([box, 0])
                        people_count += 1

                    tl = round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
                    cv2.rectangle(image, pt1=(box[0], box[1]), pt2=(box[2], box[3]),color=head_color, thickness=tl)
            

            prev_heads = [head if i in overlapped_prev else [head[0], head[1]+1] for i, head in enumerate(prev_heads) if head[1] + 1 < 20]
            prev_heads.extend(new_heads)
            


        faces = app.new_get(img, prev_result, iou_threshold) if use_prev_results else app.new_get(img, False, iou_threshold)
        # faces = app.get(img)
        # print(informations)
        curr_face_indexes = []        
        if faces:
            st = time.time()
            # print(f"current embs : {len(face_embedding_vectors)}")
            for i, face in enumerate(faces):
                # print(emb)
                bbox = face.bbox.astype(np.int32)
                # gender = face.sex
                gender_prob = face.gender
                man_condition = gender_prob[0] > gender_prob[1] if gender_model == 'caffe' else gender_prob[0] < gender_prob[1]
                # print(gender_prob)
                # print(f"Width : {bbox[2] - bbox[0]} Height : {bbox[3] - bbox[1]}")     
                if man_condition:
                    gender = 'M'
                else:
                    gender = 'F'
                age = 0 if face.age < 10 else int(str(face.age)[0] + "0")
                if face.kps is not None:
                    kps = face.kps.astype(np.int32)
                    is_looking = True if check_face_front(kps) else False
                else:
                    is_looking = False
                # age = 0 if face.age < 10 else int(str(face.age)[0] + "0")

                # bbox_width = int(bbox[2]) - int(bbox[0])
                # bbox_height = int(bbox[3]) - int(bbox[1])
                # if prev_n_face_info_dict and use_prev_results:
                if prev_n_face_info_dict:
                    # print(list(prev_n_face_info_dict.keys()))

                    iou_score_dict = {}
                    for id, info in prev_n_face_info_dict.items():
                        stored_box = info['bbox']
                        iou_score_dict[id] = iou(stored_box, bbox)
                    # print(list(iou_score_dict.values()))
                    max_iou_face = sorted(iou_score_dict, key=lambda x : iou_score_dict[x])[-1]
                    # print(i, iou_score_dict)
                    # print(max_iou_face)
                    # 직전 face bbox중 iou값이 0.4이상인 face가 있으면 해당 face 정보를 업데이트
                    # print(f"{i} : {iou_score_dict[max_iou_face]}")
                    if iou_score_dict[max_iou_face] > iou_threshold:
                        # print("iou : ", max_iou_face)
                        # print("using Prev : " , i, iou_score_dict[max_iou_face])
                        # 만약 겹친다면 똑같은 임베딩 값이 채워지므로 삭제
                        face_info_dict[max_iou_face]["bbox"] = bbox
                        face_info_dict[max_iou_face]["look_count"] += 1 if is_looking else 0
                        face_info_dict[max_iou_face]["looking"] = is_looking
                        face_info_dict[max_iou_face]["disappear_count"] = 0
                        face_info_dict[max_iou_face]["appear_count"] += 1
                        face_info_dict[max_iou_face]["age_gender"][gender][age] += 1
                        face_info_dict[max_iou_face]["gender_prob"] = gender_prob
                        curr_gender = "M" if sum(face_info_dict[max_iou_face]["age_gender"]['M'].values()) > sum(face_info_dict[max_iou_face]["age_gender"]['F'].values()) else "F"
                        face_info_dict[max_iou_face]["gender"] = curr_gender
                        face_info_dict[max_iou_face]["age"] = sorted(face_info_dict[max_iou_face]["age_gender"][curr_gender], key=lambda x : face_info_dict[max_iou_face]["age_gender"][curr_gender][x])[-1]
                        curr_age = face_info_dict[max_iou_face]["age"]
                        # curr_age = 0 if curr_age < 10 else int(str(curr_age)[0] + "0")
                        informations['look_count'][curr_gender][curr_age] += 1 if is_looking else 0
                        curr_face_indexes.append(int(max_iou_face.split('_')[-1]))
                        temp_result.append(face)
                        # num += 1
                        # print(face_info_dict[max_iou_face]["age_gender"])
                        continue             
                # print("no using face : ", i)                   
                # 기존 face bank가 비어있다면
                face_id += 1
                age_gender_dict = {"M" : {i : 0 for i in range(0, 110, 10)},
                                    "F" : {i : 0 for i in range(0, 110, 10)}}
                age_gender_dict[gender][age] += 1
                face_info_dict[f"face_{face_id}"] = {
                                                "age_gender" : age_gender_dict,                                                    
                                                "gender" : gender,
                                                "gender_prob" : gender_prob,
                                                "age" : age, 
                                                "bbox" : bbox,
                                                "disappear_count" : 0,
                                                "appear_count" : 1,
                                                "look_count" : 1 if is_looking else 0,
                                                "looking" : is_looking,
                                                "stored_time" : time.strftime("%y-%m-%d-%H-%M-%S", time.localtime())}
                curr_face_indexes.append(face_id)
                informations['person_count'][gender][age] += 1
                informations['look_count'][gender][age] += 1 if is_looking else 0
            ed = time.time()
            # print(f"process faces and indexing time : {round((ed - st) * 1000, 2)}ms")

        # else:
        #     print("No face")
                    
        # print(num)
                 
            
        # print(curr_face_indexes)

        st = time.time()
        disappeared_face = [face for face in list(face_info_dict.keys()) if int(face.split('_')[-1]) not in curr_face_indexes]
        for dis_face in disappeared_face:
            face_info_dict[dis_face]['disappear_count'] += 1

            # min_appear // 2 이상 연속으로 등장했다면 현재프레임에 없더라도 초기화하지않음
            if min_appear:
                if face_info_dict[dis_face]['appear_count'] < min_appear // 2:
                    face_info_dict[dis_face]['appear_count'] = 0
        
        for k, v in face_info_dict.items():
            # print(v['disappear_count'])
            if min_appear:
                if v['disappear_count'] >= time_out and v['appear_count'] >= min_appear:
            # if v['disappear_count'] >= time_out:
                    face_time_out[k] = v
            else:
                if v['disappear_count'] >= time_out:
                    face_time_out[k] = v                

        face_info_dict = {k:v for k,v in face_info_dict.items() if v['disappear_count'] < time_out}
        prev_n_face_info_dict = {k:v for k,v in face_info_dict.items() if v['disappear_count'] < disappear_tolerance}
        # print(prev_n_face_info_dict.keys())
        prev_face_info_dict = {k:v for k,v in face_info_dict.items() if v['disappear_count'] == 0}
        ed = time.time()
        # print(f"post process time : {round((ed - st) * 1000, 2)}ms")


        # print(prev_n_face_info_dict)
        # if face_info_dict:
        #     for id in face_info_dict.keys():
        #         print(face_info_dict[id]['disappear_count'])
        #         print(len(face_info_dict[id]['embedding']))
                
        st = time.time()
        for id, info in face_info_dict.items():
            if info['disappear_count'] == 0:
                box = list(map(lambda x : max(0, x), info['bbox']))
                color = (0, 0, 255)
                area = int((box[2] - box[0]) * (box[3] - box[1]))
                # cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(image,'%s'%(info['look_count']), (box[2]+1, box[3]-5),cv2.FONT_HERSHEY_COMPLEX, 1,(0, 255, 127),2)
                if blur_face:
                    image[box[1]:box[3], box[0]:box[2]] = cv2.blur(image[box[1]:box[3], box[0]:box[2]], (30,30))
                offset = area * 0.002 if area < 25000 else 50
                # print(area)
                if info['looking']:
                    color = (0, 255, 0)
                    text = 'Front'
                else:
                    color = (0, 0, 255)
                    text = 'No Front'

                cv2.line(image, (box[0], box[1]), (int(box[0] + offset), box[1]), color, 2)
                cv2.line(image, (box[0], box[1]), (box[0], int(box[1] + offset)), color, 2)

                cv2.line(image, (box[0], box[3]), (int(box[0] + offset), box[3]), color, 2)
                cv2.line(image, (box[0], box[3]), (box[0], int(box[3] - offset)), color, 2)

                cv2.line(image, (box[2], box[1]), (int(box[2] - offset), box[1]), color, 2)
                cv2.line(image, (box[2], box[1]), (box[2], int(box[1] + offset)), color, 2)

                cv2.line(image, (box[2], box[3]), (int(box[2] - offset), box[3]), color, 2)
                cv2.line(image, (box[2], box[3]), (box[2], int(box[3] - offset)), color, 2)
                # cv2.putText(image, text, (box[2] + 2, box[1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0),2, cv2.LINE_AA)
                # cv2.putText(image, f"W : {box[2] - box[0]} | H : {box[3] - box[1]}", (box[0] + 100, box[1]-4),cv2.FONT_HERSHEY_COMPLEX, 1,(0, 200, 0),2)
                if info['gender'] is not None and info['age'] is not None and draw_age_gender:
                    if info['gender'] == 'F':
                        cv2.putText(image,'%s,%d'%(info['gender'],info['age']), (box[0]-1, box[1]-4),cv2.FONT_HERSHEY_COMPLEX, 1,(0, 69, 255),2)                        
                        # cv2.putText(image,'%s %s,%d'%(id.split('_')[-1], info['gender'],info['age']), (box[0]-1, box[1]-4),cv2.FONT_HERSHEY_COMPLEX, 1,(0, 69, 255),2)                        
                        # cv2.putText(image,'%s %s,%d %.2f %.2f'%(id.split('_')[-1], info['gender'],info['age'], round(info['gender_prob'][0], 2), round(info['gender_prob'][1], 2)), (box[0]-1, box[1]-4),cv2.FONT_HERSHEY_COMPLEX, 1,(0, 69, 255),2)
                        # cv2.putText(image,'%.2f %.2f'%(round(info['gender_prob'][0], 2), round(info['gender_prob'][1], 2)), (box[0]-1, box[1]-4),cv2.FONT_HERSHEY_COMPLEX, 1,(0, 69, 255),2)
                    else:
                        cv2.putText(image,'%s,%d'%(info['gender'],info['age']), (box[0]-1, box[1]-4),cv2.FONT_HERSHEY_COMPLEX, 1,(200, 100, 0),2)
                        # cv2.putText(image,'%s %s,%d'%(id.split('_')[-1], info['gender'],info['age']), (box[0]-1, box[1]-4),cv2.FONT_HERSHEY_COMPLEX, 1,(200, 100, 0),2)
                        # cv2.putText(image,'%s %s,%d %.2f %.2f'%(id.split('_')[-1], info['gender'],info['age'], round(info['gender_prob'][0], 2), round(info['gender_prob'][1], 2)), (box[0]-1, box[1]-4),cv2.FONT_HERSHEY_COMPLEX, 1,(200, 100, 0),2)                        
                        # cv2.putText(image,'%.2f %.2f'%(round(info['gender_prob'][0], 2), round(info['gender_prob'][1], 2)), (box[0]-1, box[1]-4),cv2.FONT_HERSHEY_COMPLEX, 1,(200, 100, 0),2)

            # else:
            #     print(f"{id} : disappeared")
        ed = time.time()
        # print(f"draw time : {round((ed - st) * 1000, 2)}ms")

        ed_total = time.time()
        response_time = int((ed_total - st_total)*1000)
        # if response_time < 2000:
        #     total_response_time += response_time
        #     max_response_time = max(max_response_time, response_time)
        total_response_time += response_time
        max_response_time = max(max_response_time, response_time)        
        min_response_time = min(min_response_time, response_time)
        terminate_t = timeit.default_timer()
        fps = int(1./(terminate_t -  fps_start))
        max_fps = max(fps, max_fps)
        min_fps = min(fps, min_fps)
        total_fps += fps
        cv2.putText(image, f"FPS : {fps}  infer : {response_time}ms", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2, cv2.LINE_AA)
        cv2.putText(image, f"{people_count} Peoples", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2, cv2.LINE_AA)
        
        # if faces:
        #     print(f"{len(faces)} persons, infer time per person : {round(response_time / len(faces), 2)}ms")

        prev_result = temp_result
        # prev_result = faces

        if draw:
            if image.shape[0] > 1080 or image.shape[1] > 1920 and mode == 'webcam':
                image = cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_AREA)
            cv2.imshow("img", image)
        key = cv2.waitKey(1)
        if key == 27: # esc
            break

        if save_video:
            out_video.write(image)
        if mode != 'webcam':
            pbar.update()

    if mode != 'webcam':
        pbar.close()

    for k, v in face_info_dict.items():
        if min_appear:
            if v['appear_count'] >= min_appear:
                face_time_out[k] = v
        else:
            face_time_out[k] = v

    average_fps = total_fps / cnt if cnt != 0 else 0
    average_response_time = total_response_time / cnt if cnt != 0 else 0
    logging.info(f"path={path} \
                coreml={coreml} \
                face_model={face_model} \
                face_det_size={face_det_size} \
                draw={draw} \
                draw_age_gender={draw_age_gender} \
                filename={filename} \
                save_video={save_video} \
                save_log={save_log} \
                time_out={time_out} \
                disappear_tolerance={disappear_tolerance} \
                use_prev_results={use_prev_results} \
                iou_threshold={iou_threshold}")
    logging.info(f"Max FPS : {max_fps}")
    logging.info(f"Min FPS : {min_fps}")
    logging.info(f"Average FPS : {int(average_fps)}\n")
    logging.info(f"Max Response Time : {max_response_time}ms")
    logging.info(f"Min Response Time : {min_response_time}ms")
    logging.info(f"Average Response Time : {int(average_response_time)}ms")
    if mode == "webcam":
        logging.info(f"Total Frames : {cnt}\n")
    else:
        logging.info(f"Total Frames : {length}\n")

    if save_log or save_video:
        valid_keys = ['age_gender', 'look_count', 'stored_time', 'appear_count']
        records = []
        for i, (face_id, face_info) in enumerate(face_time_out.items()):
            record = [i, face_id,]
            for key in valid_keys:
                if key == 'age_gender':
                    # result = [f"{k}_{sorted(v, key = lambda x : v[x])[-1]}_{v[sorted(v, key = lambda x : v[x])[-1]]}" for k, v in face_info[key].items()]
                    # gender_idx = np.argmax([int(gender_age.split('_')[-1]) for gender_age in result])
                    # gender = result[gender_idx].split('_')[0]
                    # age = result[gender_idx].split('_')[1]
                    gender = face_info['gender']
                    age = face_info['age']
                    record.extend([gender, age])
                elif key == 'look_count':
                    watching_time = round(face_info[key] / average_fps if mode != 'video' else face_info[key] / video_fps, 2)
                    record.append(watching_time)
                else:
                    record.append(face_info[key])
            
            records.append(record)
        
        cursor.executemany(f"INSERT INTO pedestrians(id, face_id, gender, age, watching_time, stored_time, appear_count)\
                            VALUES(?,?,?,?,?,?,?)", records)
        conn.commit()
        cursor.close()
        conn.close()


    total_pedestrian_num = 0
    male_num = 0
    female_num = 0
    watch_time_list = []
    final_results = sorted(face_time_out.values(), key=lambda x : x['look_count'])
    max_watching_time = round(final_results[-1]['look_count'] / video_fps, 2) if mode == 'video' else round(final_results[-1]['look_count'] / average_fps, 2)
    max_watching_time_gender = final_results[-1]['gender']
    max_watching_time_age = final_results[-1]['age']
    max_watching_time_time = final_results[-1]['stored_time']

    logging.info(f"최장 주시 시간 : {max_watching_time}")
    logging.info(f"최장 주시 성별 : {max_watching_time_gender}")
    logging.info(f"최장 주시 나이 : {max_watching_time_age}")
    logging.info(f"최장 주시 시간대 : {max_watching_time_time}")

    # for k, v in face_time_out.items():
    #     watch_time_list.append(round(v['look_count'] / video_fps, 2))

    total_num = len(final_results)
    male_num_2 = len([face for face in final_results if face['gender'] == 'M'])
    female_num_2 = len([face for face in final_results if face['gender'] == 'F'])

    logging.info(f"총 보행자 수 : {total_num}")
    logging.info(f"총 남자 보행자 수 : {male_num_2}")
    logging.info(f"총 여자 보행자 수: {female_num_2}")    

    total_watching_time = sum([face['look_count'] for face in final_results]) / average_fps if mode != 'video' else sum([face['look_count'] for face in final_results]) / video_fps
    total_watching_time = round(total_watching_time / total_num, 2) if total_num != 0 else 0
    male_watching_time = sum([face['look_count'] for face in final_results if face['gender'] == 'M']) / average_fps if mode != 'video' else sum([face['look_count'] for face in final_results if face['gender'] == 'M']) / video_fps
    male_watching_time = round(male_watching_time / male_num_2, 2) if male_num_2 != 0 else 0
    female_watching_time = sum([face['look_count'] for face in final_results if face['gender'] == 'F']) / average_fps if mode != 'video' else sum([face['look_count'] for face in final_results if face['gender'] == 'F']) / video_fps
    female_watching_time = round(female_watching_time / female_num_2, 2) if female_num_2 != 0 else 0


    age_gender_look_count = {"M" : {i : 0 for i in range(0, 110, 10)},
                             "F" : {i : 0 for i in range(0, 110, 10)}}
    age_gender_num_count = {"M" : {i : 0 for i in range(0, 110, 10)},
                             "F" : {i : 0 for i in range(0, 110, 10)}}

    for face in final_results:
        age_gender_num_count[face['gender']][face['age']] += 1

    logging.info(f"\n연령대별 인원수")
    for gender, age_nums in age_gender_num_count.items():
        logging.info(f"남자") if gender == "M" else logging.info(f"여자")
        for ages, nums in age_nums.items():
            if ages == 0:
                logging.info(f"10대이하 : {nums}명")
            elif ages == 100:
                logging.info(f"100세 이상 : {nums}명")
            else:
                logging.info(f"{ages}대 : {nums}명")     
    
    for face in final_results:
        age_gender_look_count[face['gender']][face['age']] += face['look_count'] / average_fps if mode != 'video' else face['look_count'] / video_fps

    logging.info(f"\n주시시간")
    for (gender_look, looking_times), (gender_count, age_nums) in zip(age_gender_look_count.items(), age_gender_num_count.items()):
        logging.info(f"남자") if gender_look == "M" else logging.info(f"여자")
        for ages, times in looking_times.items():
            if ages == 0:
                logging.info(f"10대이하 : {round(times / age_nums[ages], 2) if age_nums[ages] != 0 else 0}초")
            elif ages == 100:
                logging.info(f"100세 이상 : {round(times / age_nums[ages], 2) if age_nums[ages] != 0 else 0}초")
            else:
                logging.info(f"{ages}대 : {round(times / age_nums[ages], 2) if age_nums[ages] != 0 else 0}초")                    

    logging.info(f"전체 평균 주시시간 : {total_watching_time}s")
    logging.info(f"남자 평균 주시시간 : {male_watching_time}s")
    logging.info(f"여자 평균 주시시간 : {female_watching_time}s") 



def main(filename=None,
         coreml=False,
         path='mot17-08-SDP.mov', 
         face_model='buffalo_l', 
         face_det_size=640, 
         draw=True,
         draw_age_gender=True, 
         save_video=False,
         save_log=False,
         time_out=20,
         disappear_tolerance=5,
         use_prev_results=True,
         iou_threshold=0.3,
         save_original=True,
         min_appear=20,
         gender_model='youtube',
         blur_face=False,
         min_box_size=0,
         head_detector=True):
    filename = path if not filename else filename
    video(path=path,
          coreml=coreml, 
          face_model=face_model, 
          face_det_size=face_det_size, 
          draw=draw,
          draw_age_gender=draw_age_gender, 
          filename=filename, 
          save_video=save_video,
          save_log=save_log,
          time_out=time_out,
          disappear_tolerance=disappear_tolerance,
          use_prev_results=use_prev_results,
          iou_threshold=iou_threshold,
          save_original=save_original,
          min_appear=min_appear,
          gender_model=gender_model,
          blur_face=blur_face,
          min_box_size=min_box_size,
          head_detector=head_detector)



if __name__ == '__main__':
    fire.Fire(main)
