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
from utils3 import check_face_front, letterbox
import torch
import torch.nn as nn
import sqlite3
from centroid_tracker import EuclideanDistTracker
import math
from datetime import datetime
import pandas as pd
import smtplib
from email import encoders
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase


def preprocess(img):
    img_head = img.copy()
    img_head, ratio, dwdh = letterbox(img_head, auto=False)
    img_head = img_head.transpose((2, 0, 1))
    img_head = np.expand_dims(img_head, 0)
    img_head = np.ascontiguousarray(img_head)

    im = img_head.astype(np.float32)
    im /= 255
    return ratio,dwdh,im  

def make_xlsx(db_path, total_count):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT * FROM pedestrians")
    output = cur.fetchall()
    df = pd.DataFrame(output)
    xlsx_path = os.path.splitext(db_path)[0] + f'-{total_count}_pedes.xlsx'
    df.to_excel(xlsx_path)
    return xlsx_path

def send_email(db_path, total_count):
    # xlsx_path = make_xlsx(db_path, total_count)
    filename = os.path.basename(db_path)
    attach_file = open(db_path, "rb")

    smtp = smtplib.SMTP('smtp.gmail.com', 587)
    smtp.ehlo()
    smtp.starttls()
    smtp.login('knh@xinapse.ai', 'yqsyxyhzmrftffte')

    date = datetime.now().strftime('Analysis Result %Y.%m.%d - %H:%M:%S')
    msg = MIMEMultipart()
    msg['Subject'] = date
    file_data = MIMEBase("application", "octet-stream")
    file_data.set_payload(attach_file.read())
    encoders.encode_base64(file_data)
    file_data.add_header("Content-Disposition", "attachment", filename = filename)
    msg.attach(file_data)

    smtp.sendmail('knh@xinapse.ai', 'ejrtks102020@gmail.com', msg.as_string())
    smtp.quit()    


def video(path,
          face_model,
          face_det_size,
          draw,
          draw_age_gender,
          filename,
          save_video,
          save_log,
          disappear_tolerance,
          save_original,
          min_appear,
          gender_model,
          blur_face,
          min_box_size,
          head_detector):
    providers = ['CPUExecutionProvider']
    if head_detector:
        from openvino.runtime import Core
        model_path = "yolo-head-openvino/yolo-tiny-head.xml"
        ie = Core()
        network = ie.read_model(model=model_path, weights="yolo-head-openvino/yolo-tiny-head.bin")
        executable_network = ie.compile_model(model=network, device_name="CPU")
        output_layer = next(iter(executable_network.outputs))
    if gender_model == 'aihub':
        app = FaceAnalysis(name=face_model, allowed_modules=['detection'], 
                           providers=providers, gender=gender_model, root='.',
                           min_box_size=min_box_size)
    else:
        app = FaceAnalysis(name=face_model, allowed_modules=['detection', 'genderage'],
                           providers=providers, gender=gender_model, root='.',
                           min_box_size=min_box_size) 
                        
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

    db_path = None
    if save_log or save_video:
        db_path = f'insightface_output/{filename}/{filename}-{start_datatime}.db'
        conn = sqlite3.connect(db_path)
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

    tracker = EuclideanDistTracker(disappear_tolerance=disappear_tolerance)
    face_id = 0

    prev_result = []
    head_color = [random.randint(0, 255) for _ in range(3)]
    face_time_out = {}
    if mode != 'webcam':
        pbar = tqdm.tqdm(total=length, position=0, leave=True)
    
    isDrawing = False
    no_display = np.zeros((1, 1))
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
        
        if head_detector:
            ratio, dwdh, im = preprocess(img)
            y = executable_network([im])[output_layer]
            heads = []
            for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(y):
            	if cls_id == 0:
                    box = np.array([x0,y0,x1,y1])
                    box -= np.array(dwdh*2)
                    box /= ratio
                    x0, y0, x1, y1 = box.round().astype(np.int32).tolist()
                    head_area = (x1 - x0) * (y1 - y0)
                    # print(head_area)
                    if head_area < 5000:
                        continue

                    heads.append([x0, y0, x1, y1])

            boxes_ids = tracker.update(heads)
            for box_id in boxes_ids:
                x0, y0, x1, y1, id = box_id
                tl = round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1
                cv2.putText(image, str(id),(x0,y0-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                cv2.rectangle(image, pt1=(x0, y0), pt2=(x1, y1),color=head_color, thickness=tl)
            cv2.putText(image, f"{tracker.id_count} Peoples", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2, cv2.LINE_AA)
            # print(tracker.id_count)       
            
        faces = app.new_get(img)

        curr_face_indexes = []        
        objects_bbs_ids = []
        if faces:
            st = time.time()
            for i, face in enumerate(faces):
                bbox = face.bbox.astype(np.int32)
                x1, y1, x2, y2 = bbox[:4]
                gender_prob = face.gender
                man_condition = gender_prob[0] > gender_prob[1] if gender_model == 'caffe' else gender_prob[0] < gender_prob[1]
                gender = 'M' if man_condition else 'F'
                age = 0 if face.age < 10 else int(str(face.age)[0] + "0")
                if face.kps is not None:
                    kps = face.kps.astype(np.int32)
                    is_looking = True if check_face_front(kps) else False
                else:
                    is_looking = False
                center_x = (bbox[0] + bbox[2]) // 2
                center_y = (bbox[1] + bbox[3]) // 2
                same_face_detected = False
                if face_info_dict:
                    # print(list(prev_n_face_info_dict.keys()))
                    dists = [ (id, math.hypot(center_x - face_info["xy"][0], center_y - face_info["xy"][1])) for id, face_info in face_info_dict.items() ]
                    dists = sorted(dists, key=lambda x : x[1])
                    for id, dist in dists:
                        if id in curr_face_indexes:
                            continue

                        if dist < 500:
                            face_info_dict[id]["bbox"] = bbox
                            face_info_dict[id]["xy"] = (center_x, center_y)
                            face_info_dict[id]["look_count"] += 1 if is_looking else 0
                            face_info_dict[id]["looking"] = is_looking
                            face_info_dict[id]["disappear_count"] = 0
                            face_info_dict[id]["appear_count"] += 1
                            face_info_dict[id]["age_gender"][gender][age] += 1
                            face_info_dict[id]["gender_prob"] = gender_prob
                            curr_gender = "M" if sum(face_info_dict[id]["age_gender"]['M'].values()) > sum(face_info_dict[id]["age_gender"]['F'].values()) else "F"
                            face_info_dict[id]["gender"] = curr_gender
                            face_info_dict[id]["age"] = sorted(face_info_dict[id]["age_gender"][curr_gender], key=lambda x : face_info_dict[id]["age_gender"][curr_gender][x])[-1]
                            curr_age = face_info_dict[id]["age"]
                            informations['look_count'][curr_gender][curr_age] += 1 if is_looking else 0
                            objects_bbs_ids.append([x1,y1,x2,y2,id])
                            curr_face_indexes.append(id)
                            same_face_detected = True
                        break

                if not same_face_detected:
                    age_gender_dict = {"M" : {i : 0 for i in range(0, 110, 10)},
                                        "F" : {i : 0 for i in range(0, 110, 10)}}
                    age_gender_dict[gender][age] += 1
                    face_info_dict[face_id] = {
                                                    "age_gender" : age_gender_dict,                                                    
                                                    "gender" : gender,
                                                    "gender_prob" : gender_prob,
                                                    "age" : age, 
                                                    "bbox" : bbox,
                                                    "xy" : (center_x, center_y),
                                                    "disappear_count" : 0,
                                                    "appear_count" : 1,
                                                    "look_count" : 1 if is_looking else 0,
                                                    "looking" : is_looking,
                                                    "stored_time" : time.strftime("%y-%m-%d-%H-%M-%S", time.localtime())}
                    informations['person_count'][gender][age] += 1
                    informations['look_count'][gender][age] += 1 if is_looking else 0
                    objects_bbs_ids.append([x1,y1,x2,y2,face_id])
                    curr_face_indexes.append(face_id)
                    face_id += 1

            ed = time.time()
            # print(f"process faces and indexing time : {round((ed - st) * 1000, 2)}ms")
        current_ids = [bb_id[-1] for bb_id in objects_bbs_ids]
        new_face_info_dict = {}
        for id, face_info in face_info_dict.items():
            if id in current_ids:
                new_face_info_dict[id] = face_info
            elif face_info["disappear_count"] <= disappear_tolerance:
                new_face_info_dict[id] = face_info
                new_face_info_dict[id]["disappear_count"] += 1
            elif face_info["disappear_count"] > disappear_tolerance:
                face_time_out[id] = face_info
        
        face_info_dict = new_face_info_dict.copy()
                
        st = time.time()
        for id, info in face_info_dict.items():
            if info['disappear_count'] == 0:
                box = list(map(lambda x : max(0, x), info['bbox']))
                color = (0, 0, 255)
                area = int((box[2] - box[0]) * (box[3] - box[1]))
                # cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(image,'%s'%(info['look_count']), (box[2]+1, box[3]-5),cv2.FONT_HERSHEY_COMPLEX, 1,(0, 255, 127),2)
                cv2.putText(image, f"Face_{id}", (box[0]+50, box[1]-4),cv2.FONT_HERSHEY_COMPLEX, 1,(0, 255, 0),2)

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
        # cv2.putText(image, f"{people_count} Peoples", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2, cv2.LINE_AA)
        
        # if faces:
        #     print(f"{len(faces)} persons, infer time per person : {round(response_time / len(faces), 2)}ms")

        prev_result = temp_result
        # prev_result = faces

        if draw:
            if isDrawing:
                if (image.shape[0] > 1080 or image.shape[1] > 1920) or mode == 'webcam':
                    image = cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_AREA)
                cv2.imshow("CAM", image)
            else:
                cv2.imshow("CAM", no_display)
        key = cv2.waitKey(1)
        
        if key & 0xFF == ord('d'): # esc
            isDrawing = not isDrawing

        if key & 0xFF == ord('x'):
            break	    

        if save_video:
            out_video.write(image)
        if mode != 'webcam':
            pbar.update()
        
        # 오전 00시 50분 카메라 종료
        if datetime.now().strftime('%H:%M') == "00:50":
            break        

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
                face_model={face_model} \
                face_det_size={face_det_size} \
                draw={draw} \
                draw_age_gender={draw_age_gender} \
                filename={filename} \
                save_video={save_video} \
                save_log={save_log} \
                disappear_tolerance={disappear_tolerance}")
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

    if db_path != None:
        send_email(db_path, tracker.id_count)

def main(filename=None,
         path='webcam', 
         face_model='buffalo_m', 
         face_det_size=640, 
         draw=True,
         draw_age_gender=True, 
         save_video=False,
         save_log=False,
         disappear_tolerance=20,
         save_original=True,
         min_appear=20,
         gender_model='youtube',
         blur_face=False,
         min_box_size=0,
         head_detector=True):
    filename = path if not filename else filename
    video(path=path,
          face_model=face_model, 
          face_det_size=face_det_size, 
          draw=draw,
          draw_age_gender=draw_age_gender, 
          filename=filename, 
          save_video=save_video,
          save_log=save_log,
          disappear_tolerance=disappear_tolerance,
          save_original=save_original,
          min_appear=min_appear,
          gender_model=gender_model,
          blur_face=blur_face,
          min_box_size=min_box_size,
          head_detector=head_detector)



if __name__ == '__main__':
    fire.Fire(main)
