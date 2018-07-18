import cv2
import numpy as np


def draw_rect(frame, face_rect, col=(0, 255, 0)):
    # 画出人脸区域的
    x, y, w, h = face_rect
    frame_face = frame[y:y + h, x:x + w]
    # cv2.rectangle(frame, (x, y), (x + w, y + h), col, 2)
    return frame_face


# caffe model
def detector(net, frame):
    """
    :param net:face detection model
    :param frame:frame images obtained from a webcam
    :return:3D-ndarray, face area
    """
    try:
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                    (300, 300), (104.0, 177.0, 123.0))
    except Exception as e:
        print("frame or model failed to {}".format(str(e)))

    net.setInput(blob)
    detections = net.forward()
    max_face_rect = None
    max_face_area = 0
    if detections.shape[2] != 0:
        for i in range(0, detections.shape[2]):

            confidence = detections[0, 0, i, 2]

            if confidence < 0.85:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            if startX < 0 or startY < 0 or endX > w or endY > h:
                continue
            face_area = (endX - startX) * (endY - startY)
            if face_area > max_face_area:
                max_face_rect = [startX, startY, endX, endY]
                max_face_area = face_area
        return max_face_rect


# opencv 自带的人脸检测
def detector3(net, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = net.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE)
    max_face_rect = None
    max_face_area = 0
    if len(faces) != 0:
        for face in faces:
            face_area = face[2] * face[3]
            if face_area > max_face_area:
                max_face_rect = face
                max_face_area = face_area
        return max_face_rect


