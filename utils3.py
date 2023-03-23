import cv2
import numpy as np

def cosine_similarity(vector,matrix):
   return matrix.dot(vector)/ (np.linalg.norm(matrix, axis=1) * np.linalg.norm(vector))


def draw_info(image_original, bbox, id, gender, age):
    image = image_original.copy()
    x,y,w,h = bbox
    cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)),
                color=(0, 255, 0), thickness=2)
    cv2.putText(image, f'{id}', (int(x + w*1.01), y),
                fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 0, 255), thickness=1)
    cv2.putText(image, f'{gender}', (int(x + w*1.01), int(y + h*0.3)),
                fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 0, 255), thickness=1)
    cv2.putText(image, f'{age}', (int(x + w*1.01), int(y + h*0.6)),
                fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 0, 255), thickness=1)
    return image

def iou(box1, box2):
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


def check_face_front(kps):
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
    eye_max_y = max(left_eye_y, right_eye_y)
    mouse_center_x = (left_mouse_x + right_mouse_x) // 2
    mouse_center_y = (left_mouse_y + right_mouse_y) // 2
    mouse_min_y = min(left_mouse_y, right_mouse_y)
    check_nose_x = nose_x > right_eye_x and nose_x < left_eye_x
    # check_nose_y = nose_y < mouse_center_y and nose_y > eye_center_y
    check_nose_y = nose_y < mouse_min_y and nose_y > eye_max_y
    # print(check_nose_x, check_nose_y)
    if check_nose_x and check_nose_y:
        return True
    else:
        return False    

def nms(self, dets):
    thresh = self.nms_thresh
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def cv2_letterbox_resize(img, expected_size):
    ih, iw = img.shape[0:2]
    ew, eh = expected_size
    scale = min(eh / ih, ew / iw)
    nh = int(ih * scale)
    nw = int(iw * scale)
    smat = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], np.float32)
    top = (eh - nh) // 2
    bottom = eh - nh - top
    left = (ew - nw) // 2
    right = ew - nw - left
    tmat = np.array([[1, 0, left], [0, 1, top], [0, 0, 1]], np.float32)
    amat = np.dot(tmat, smat)
    amat_ = amat[:2, :]
    dst = cv2.warpAffine(img, amat_, expected_size)
    if dst.ndim == 2:
        dst = np.expand_dims(dst, axis=-1)
    return dst, amat


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)