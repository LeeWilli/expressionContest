from __future__ import division, absolute_import
import tflearn
from os.path import isfile, join
import random
import sys


class EMR:

    def __init__(self):
        self.target_classes = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

    def build_network(self):
        """
        Build the convnet.
        Input is 48x48
        3072 nodes in fully connected layer
        """
        # Real-time data preprocessing
        img_prep = tflearn.ImagePreprocessing()
        img_prep.add_featurewise_zero_center(per_channel=True, mean=[0.53990436, 0.4405486, 0.39328504])

        # Real-time data augmentation
        img_aug = tflearn.ImageAugmentation()
        img_aug.add_random_flip_leftright()
        img_aug.add_random_crop([49, 49], padding=4)

        # Building Residual Network
        self.network = tflearn.input_data(shape=[None, 49, 49, 3],
                                          data_preprocessing=img_prep,
                                          data_augmentation=img_aug)
        self.network = tflearn.conv_2d(self.network, 16, 3, regularizer='L2', weight_decay=0.0001)
        self.network = tflearn.resnext_block(self.network, 5, 16, 32)
        self.network = tflearn.resnext_block(self.network, 1, 32, 32, downsample=True)
        self.network = tflearn.resnext_block(self.network, 4, 32, 32)
        self.network = tflearn.resnext_block(self.network, 1, 64, 32, downsample=True)
        self.network = tflearn.resnext_block(self.network, 4, 64, 32)
        self.network = tflearn.batch_normalization(self.network)
        self.network = tflearn.activation(self.network, 'relu')
        self.network = tflearn.global_avg_pool(self.network)
        # Regression
        self.network = tflearn.fully_connected(self.network, 11, activation='softmax')
        opt = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=32000, staircase=True)
        self.network = tflearn.regression(self.network, optimizer=opt,
                                          loss='categorical_crossentropy')
        # Training
        self.model = tflearn.DNN(self.network, checkpoint_path='Snapshots/model_resnext',
                                 max_checkpoints=10, tensorboard_verbose=0, tensorboard_dir='Logs/',
                                 clip_gradients=0.)
        self.load_model()

    def predict(self, image):
        """
        Image is resized to 48x48
        model.predict() is an inbuilt function in tflearn.
        """
        if image is None:
            return None
        image = image.reshape([-1, 49, 49, 3])
        return self.model.predict(image)

    def load_model(self):
        """
        Loads pre-trained model.
        model.load() is an inbuilt function in tflearn
        """
        if isfile("models/model_resnext/model_resnext-96000.meta"):
            self.model.load("models/model_resnext/model_resnext-96000")
            print('\n---> Pre-trained model loaded')
        else:
            print("---> Couldn't find model")


if __name__ == "__main__":
    print("\n------------Emotion Detection Program------------\n")
    network = EMR()
    if sys.argv[1] == 'singleface':
        import singleface
    if sys.argv[1] == 'multiface':
        import multiface
