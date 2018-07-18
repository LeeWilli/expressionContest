import builtins
import time

import cv2
import numpy as np

from utils.rect_util import Rect
import utils.img_util as imgu

result = np.zeros((1, 8))


def load_image(frame_face):
    image = cv2.cvtColor(frame_face, cv2.COLOR_BGR2GRAY)
    roi = [0, 0, image.shape[1], image.shape[0]]
    face_rc = Rect(roi)
    width = 64
    height = 64
    max_shift = 0.08
    max_scale = 1.05
    max_angle = 20.0
    max_skew = 0.05
    do_flip = True

    A, A_pinv = imgu.compute_norm_mat(width, height)

    distorted_image = imgu.distort_img(image, face_rc, width, height,
                                       max_shift, max_scale,
                                       max_angle,
                                       max_skew,
                                       do_flip)

    final_image = imgu.preproc_img(distorted_image, A=A, A_pinv=A_pinv)
    final_image = final_image.reshape(1, 4096).astype("float32")
    return final_image


def detector(frame, face_rect, models):
    '''
    :param frame: the image to be detected
    :param face_rect:face area
    :param models:micro expression model
    :return:confidence of 8 kinds of expressions
    '''

    frame_face = frame[face_rect[1]:face_rect[3], face_rect[0]:face_rect[2]]
    predictions = models(
        {"x": load_image(frame_face)})

    return predictions["probabilities"]


def detector1(frame, face_rect, models):
    try:
        frame_face = frame[face_rect[1]:face_rect[3], face_rect[0]:face_rect[2]]
    except Exception as e:
        print("frame_face failed to {}".format(str(e)))
    # frame_face = cv2.cvtColor(frame_face, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(frame_face, (64, 64))
    image = np.reshape(image, (1, 64, 64, 3))
    try:
        preds = models.predict(image)
    except Exception as e:
        print("emotion return failed to {}".format(str(e)))
    result = preds.tolist()[0]
    return result


def detector2(frame, face_rect, network):
    frame_face = frame[face_rect[1]:face_rect[3], face_rect[0]:face_rect[2]]
    image = cv2.resize(frame_face, (49, 49), interpolation=cv2.INTER_CUBIC) / 255.
    result = network.predict(image)
    return result[0][:8]


em_dat_buffer = []


def em_detection(frame, face_rect, models):
    ret = np.zeros((1, 8))
    emotion_data = []
    global em_dat_buffer

    try:
        print("detection emotion")
        result = detector(frame, face_rect, models)
        print(result)
        em_dat_buffer.append(result)
        L = len(em_dat_buffer)
        print(L)
        buffer_size = 5
        if L >= buffer_size:
            em_dat_buffer = em_dat_buffer[-buffer_size:]
            L = buffer_size
        samples = np.array(em_dat_buffer)
        emotion_data.append(np.mean(samples, axis=0))
    except Exception as e:
        print(str(e))
        return ret
    ret = emotion_data[0]
    return ret
