import os
import sys
import time

import numpy as np
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import argparse
import imutils
import cv2
from src import face_detection
from src import emotion_detection
from tensorflow.contrib.predictor import predictor_factories
from src import resnext
from keras.models import load_model

from src.em_dodel import EMR

emotion_zh_dict = {0: u'平和', 1: u'高兴', 2: u'轻蔑', 3: u'厌恶', 4: u'惊讶', 5: u'伤心',
                   6: u'害怕', 7: u'生气'}
emotion_en_dict = {0: u'neutral', 1: u'happy', 2: u'contempt', 3: u'disgusted',
                   4: u'surprised', 5: u'sad',
                   6: u'fearful', 7: u'angry'}
EMOTION = [u'neutral', u'happiness', u'surprise', u'sadness', u'anger', u'disgust', u'fear', u'contempt']
AFFECTNET = [u'neutral', u'happy', u'sad', u'surprise', u'fear', u'disgust', u'anger', u'contempt', u'none',
             u'uncertain', u'non-Face']


def format_image():
    pass


if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    ap.add_argument("-u", "--webcam",
                    default="rtsp://admin:1234qwer@192.168.16.202/h264/ch1/main/",
                    # default="rtmp://live.hkstv.hk.lxdns.com/live/hks",
                    # default="rtmp://192.168.16.122:1935/rtmplive/testzz1",
                    # default=0,
                    help="# camera url")
    ap.add_argument("-d", "--display", type=int, default=1,
                    help="Whether or not frames should be displayed")
    ap.add_argument("-p", "--prototxt", required=False,
                    default="models/face/deploy.prototxt.txt",
                    help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--face_model", required=False,
                    default="models/face/res10_300x300_ssd_iter_140000.caffemodel",
                    help="path to Caffe pre-trained model")

    ap.add_argument("-e", "--emotion_model_dir", required=False,
                    default="models/emotion/",
                    help="emotion model of dir")
    args = vars(ap.parse_args())


    def resource_path(relative_path):
        """Convert relative resource path to the real path"""
        if hasattr(sys, "_MEIPASS"):
            return os.path.join(sys._MEIPASS, relative_path)
        return os.path.join(relative_path)


    '''caffe DNN model for face detection'''
    # net = cv2.dnn.readNetFromCaffe(resource_path(args["prototxt"]), resource_path(args["face_model"]))

    '''caffe CascadeClassifier model for face detection'''
    # net = cv2.CascadeClassifier('models/face/haarcascade_frontalface_alt.xml')

    ''''tflearn emotion model'''
    # network = EMR()
    # network.build_network()

    '''keras emotion model'''
    # network = resnext.ResNextImageNet(include_top=True, input_shape=(64, 64, 1), classes=8)
    # network.load_s("models/model.h5")
    network = load_model(resource_path("models/resnet50_final.h5"))

    '''tensorflow emotion model'''
    # network = predictor_factories.from_saved_model(args["emotion_model_dir"])

    print("[INFO] sampling frames from webcam...")

    root = "F:/test"
    result_root = "F:/test_result"
    acc = []
    for lists in os.listdir(root):
        emotion_dir = os.path.join(root, lists)
        print(emotion_dir)
        right = []
        for file in os.listdir(emotion_dir):
            image_path = os.path.join(emotion_dir, file)
            new_name = file.split(".")[0]
            img = cv2.imread(image_path)
            # face_rect = face_detection.detector(net, img)

            # if face_rect is not None:
            #     cv2.rectangle(img, (face_rect[0], face_rect[1]), (face_rect[2], face_rect[3]), (0, 255, 0), 2)
            #     cv2.putText(img, "{}".format(EMOTION[np.argmax(result)]), (10, 50),
            #             cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            result = emotion_detection.detector1(img, [0, 0, 48, 48], network)

            if emotion_dir == EMOTION[np.argmax(result)]:
                right.append(1)
            else:
                right.append(0)
            cv2.imwrite(os.path.join(result_root, "{}_{}_{}.png".format(EMOTION[np.argmax(emotion_dir)], new_name,
                                                                          EMOTION[np.argmax(result)])), img)

        print("{} of emotion accuracy is {} %".format(EMOTION[np.argmax(emotion_dir)],sum(right) / len(right)))
        acc.append(sum(right) / len(right))


