Requirements:
---------------

- [Python v2.7 or v3.5+)](http://python.org/)
- [OpenCV v2+](http://opencv.org/)
- Numpy, Scipy, Scikit-learn,imutils

Project Structure:
------------
```tree
expressionContest
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
- run train.py to train the emotion model

```
python train.py \
 --num_epochs 200 'Number of training epochs'\
 --batch_size 428 'Batch size'\
 --learning_rate 1e-04 'Learning rate'\
 --train_dataset '/data/zl/AffectNet/Manually_Annotated_Images/face/training/train.tfrecords'\
 --eval_dataset '/data/zl/AffectNet/Manually_Annotated_Images/face/validation/validation.tfrecords'\
 --model_dir 'models/emotion_cnn_model'

```

Model download address:
---------------------

192.168.16.123/Share/emotion_model 


Database download address:
---------
192.168.16.166:/data/zl/EmotionNet

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
