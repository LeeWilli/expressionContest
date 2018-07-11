Requirements:
---------------

- [Python v2.7 or v3.5+)](http://python.org/)
- [OpenCV v2+](http://opencv.org/)
- Numpy, Scipy, Scikit-learn,imutils

Project Structure:
------------
```tree
webcam-pulse
│  .gitignore
│  main.py                                             # program main function
│  README.md
│  requirements.txt                                    # python dependency package
│
└─src
   │  face_detection.py                                # face detection interface
   └─ emotion_detection.py                             # emotion detection interface

```

Quickstart:
------------

- run main.py to start the application

```
python main.py\
 --webcamera "rtsp://admin:1234qwer@192.168.16.202/h264/ch1/main/"\   # camera url
 --display 1\  # Whether or not frames should be displayed:1 is display,0 is not display
 --prototxt  "resources/deploy.prototxt.txt"\ # path to Caffe deploy prototxt file
 --model "resources/res10_300x300_ssd_iter_140000.caffemodel" # path to Caffe pre-trained model
```
Module:
--------

- src/face_detection.py

```python
def detector(net, frame):
    """
    :param net:face detection model
    :param frame:frame images obtained from a webcam
    :return:3D-ndarray, face area
    """
```
- src/emotion_detection.py

```python
def emotion_detection(frame, face_rect, models):
        '''
        :param frame: the image to be detected
        :param face_rect:face area
        :param models:micro expression model
        :return:confidence of 8 kinds of expressions
        '''
```
