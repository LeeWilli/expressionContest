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

    '''caffe DNN model for face detection'''
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["face_model"])

    '''caffe CascadeClassifier model for face detection'''
    # net = cv2.CascadeClassifier('models/face/haarcascade_frontalface_alt.xml')

    ''''tflearn emotion model'''
    # network = EMR()
    # network.build_network()

    '''keras emotion model'''
    # network = resnext.ResNextImageNet(include_top=True, input_shape=(64, 64, 1), classes=8)
    # network.load_s("models/model.h5")
    network=load_model("models/resnet50_final.h5")

    '''tensorflow emotion model'''
    # network = predictor_factories.from_saved_model(args["emotion_model_dir"])

    print("[INFO] sampling frames from webcam...")
    vs = WebcamVideoStream(args["webcam"]).start()
    fps = FPS().start()

    heart_rate = None
    while True:
        time0 = time.time()
        frame = vs.read()
        frame = imutils.resize(frame, width=720)
        face_rect = face_detection.detector(net, frame)
        if face_rect is not None:
            result = emotion_detection.detector1(frame, face_rect, network)
            cv2.rectangle(frame, (face_rect[0], face_rect[1]), (face_rect[2], face_rect[3]), (0, 255, 0), 2)
            cv2.putText(frame, "{}".format(EMOTION[np.argmax(result)]), (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            if args["display"] > 0:
                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
            print("time:", time.time() - time0)
        fps.update()
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    cv2.destroyAllWindows()
    vs.stop()
