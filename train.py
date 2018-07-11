'''
Using tf.estimator and tf.data to train a cnn model in TensorFlow 1.4.

GitHub: https://github.com/secsilm/understaing-datasets-estimators-tfrecords
Chinese blog:
'''
import tensorflow as tf
import os
import json

tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags
flags.DEFINE_integer('num_epochs', 200, 'Number of training epochs')
flags.DEFINE_integer('batch_size', 1024, 'Batch size')
flags.DEFINE_float('learning_rate', 1e-12, 'Learning rate')
flags.DEFINE_float('dropout_rate', 0.5, 'Dropout rate')
flags.DEFINE_string('train_dataset', '/data/zl/AffectNet/Manually_Annotated_Images/face/training/train.tfrecords',
                    'Filename of training dataset')
flags.DEFINE_string('eval_dataset', '/data/zl/AffectNet/Manually_Annotated_Images/face/validation/validation.tfrecords',
                    'Filename of evaluation dataset')
flags.DEFINE_string('model_dir', 'models/emotion_cnn_model_test',
                    'Filename of testing dataset')
FLAGS = flags.FLAGS


def cnn_model_fn(features, labels, mode):
    # Input Layer
    input_height, input_width = 64, 64
    input_channels = 3
    input_layer = tf.reshape(features["x"],
                             [-1, input_height, input_width, input_channels], name="input")
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.2)

    # Convolutional Layer #1 and Pooling Layer #1
    conv1_1 = tf.layers.conv2d(inputs=input_layer, filters=64,
                               kernel_size=[3, 3], padding="same",
                               activation=tf.nn.relu,
                               kernel_regularizer=regularizer)
    conv1_2 = tf.layers.conv2d(inputs=conv1_1, filters=64, kernel_size=[3, 3],
                               padding="same", activation=tf.nn.relu,
                               kernel_regularizer=regularizer)
    pool1 = tf.layers.max_pooling2d(inputs=conv1_2, pool_size=[2, 2],
                                    strides=2, padding="same")
    dropout1 = tf.layers.dropout(inputs=pool1, rate=0.25,
                                 training=mode == tf.estimator.ModeKeys.TRAIN)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2_1 = tf.layers.conv2d(inputs=dropout1, filters=128,
                               kernel_size=[3, 3], padding="same",
                               activation=tf.nn.relu,
                               kernel_regularizer=regularizer)
    conv2_2 = tf.layers.conv2d(inputs=conv2_1, filters=128, kernel_size=[3, 3],
                               padding="same", activation=tf.nn.relu,
                               kernel_regularizer=regularizer)
    pool2 = tf.layers.max_pooling2d(inputs=conv2_2, pool_size=[2, 2],
                                    strides=2, padding="same")
    dropout2 = tf.layers.dropout(inputs=pool2, rate=0.25,
                                 training=mode == tf.estimator.ModeKeys.TRAIN)

    # Convolutional Layer #3 and Pooling Layer #3
    conv3_1 = tf.layers.conv2d(inputs=dropout2, filters=256,
                               kernel_size=[3, 3], padding="same",
                               activation=tf.nn.relu,
                               kernel_regularizer=regularizer)
    conv3_2 = tf.layers.conv2d(inputs=conv3_1, filters=256, kernel_size=[3, 3],
                               padding="same", activation=tf.nn.relu,
                               kernel_regularizer=regularizer)
    conv3_3 = tf.layers.conv2d(inputs=conv3_2, filters=256, kernel_size=[3, 3],
                               padding="same", activation=tf.nn.relu,
                               kernel_regularizer=regularizer)
    pool3 = tf.layers.max_pooling2d(inputs=conv3_3, pool_size=[2, 2],
                                    strides=2, padding="same")
    dropout3 = tf.layers.dropout(inputs=pool3, rate=0.25,
                                 training=mode == tf.estimator.ModeKeys.TRAIN)

    # Convolutional Layer #4 and Pooling Layer #4
    conv4_1 = tf.layers.conv2d(inputs=dropout3, filters=256,
                               kernel_size=[3, 3], padding="same",
                               activation=tf.nn.relu)
    conv4_2 = tf.layers.conv2d(inputs=conv4_1, filters=256, kernel_size=[3, 3],
                               padding="same", activation=tf.nn.relu,
                               kernel_regularizer=regularizer)
    conv4_3 = tf.layers.conv2d(inputs=conv4_2, filters=256, kernel_size=[3, 3],
                               padding="same", activation=tf.nn.relu,
                               kernel_regularizer=regularizer)
    pool4 = tf.layers.max_pooling2d(inputs=conv4_3, pool_size=[2, 2],
                                    strides=2, padding="same")
    dropout4 = tf.layers.dropout(inputs=pool4, rate=0.25,
                                 training=mode == tf.estimator.ModeKeys.TRAIN)
    # # Convolutional Layer #5 and Pooling Layer #5
    # conv5_1 = tf.layers.conv2d(inputs=pool4, filters=512, kernel_size=[3, 3],
    #  padding="same", activation=tf.nn.relu)
    # conv5_2 = tf.layers.conv2d(inputs=conv5_1, filters=512, kernel_
    # size=[3, 3], padding="same", activation=tf.nn.relu)
    # pool5 = tf.layers.max_pooling2d(inputs=conv5_2, pool_size=[2, 2],
    #  strides=2, padding="same")

    # FC Layers
    pool5_flat = tf.layers.flatten(dropout4)
    FC1 = tf.layers.dense(inputs=pool5_flat, units=1024, activation=tf.nn.relu,
                          kernel_regularizer=regularizer)
    dropout5 = tf.layers.dropout(inputs=FC1, rate=0.5,
                                 training=mode == tf.estimator.ModeKeys.TRAIN)
    FC2 = tf.layers.dense(inputs=dropout5, units=1024, activation=tf.nn.relu,
                          kernel_regularizer=regularizer)
    dropout6 = tf.layers.dropout(inputs=FC2, rate=0.5,
                                 training=mode == tf.estimator.ModeKeys.TRAIN)

    """the training argument takes a boolean specifying whether or not the model is currently 
    being run in training mode; dropout will only be performed if training is true. here, 
    we check if the mode passed to our model function cnn_model_fn is train mode. """

    # Logits Layer or the output layer. which will return the raw values for our predictions.
    # Like FC layer, logits layer is another dense layer. We leave the activation function empty
    # so we can apply the softmax
    logits = tf.layers.dense(inputs=dropout6, units=8, name="output")

    # Predicition
    predictions = {
        'classes': tf.argmax(input=logits, axis=1, name='classes'),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Loss for train and eval
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=8)
    # print('onehot_labels', onehot_labels.shape)
    loss = tf.losses.softmax_cross_entropy(onehot_labels, logits, scope='LOSS')
    # print(labels.shape, predictions['classes'].shape)

    accuracy, update_op = tf.metrics.accuracy(
        labels=labels, predictions=predictions['classes'], name='accuracy')
    batch_acc = tf.reduce_mean(tf.cast(
        tf.equal(tf.cast(labels, tf.int64), predictions['classes']),
        tf.float32))
    tf.summary.scalar('batch_acc', batch_acc)
    tf.summary.scalar('streaming_acc', update_op)

    # Train
    if mode == tf.estimator.ModeKeys.TRAIN:
        # tensors_to_log = {
        #     'Accuracy': accuracy,
        #     'My accuracy': batch_acc}
        # logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,
        #  every_n_iter=100)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,
                                                   tf.train.get_global_step(),
                                                   decay_steps=20000,
                                                   decay_rate=0.8, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(
                loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                          train_op=train_op)

    eval_metric_ops = {
        'accuracy': (accuracy, update_op)
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                      eval_metric_ops=eval_metric_ops)


def parser(record):
    keys_to_features = {
        'img_raw': tf.FixedLenFeature((), tf.string),
        'label': tf.FixedLenFeature((), tf.int64),
        'width': tf.FixedLenFeature((), tf.int64),
        'height': tf.FixedLenFeature((), tf.int64),
    }
    parsed = tf.parse_single_example(record, keys_to_features)
    image = tf.decode_raw(parsed['img_raw'], tf.uint8)
    width = tf.cast(parsed['width'], tf.int32)
    height = tf.cast(parsed['height'], tf.int32)
    image = tf.reshape(image, [width, height, 3])
    image = tf.image.resize_images(image, (64, 64))
    image = tf.image.convert_image_dtype(image, tf.float32)
    # image = tf.image.random_flip_left_right(image)
    # image = tf.image.random_flip_up_down(image)
    label = tf.cast(parsed['label'], tf.int32)
    return {"x": image}, label


def save_hp_to_json():
    '''Save hyperparameters to a json file'''
    filename = os.path.join(FLAGS.model_dir, 'hparams.json')
    hparams = FLAGS.flag_values_dict()
    with open(filename, 'w') as f:
        json.dump(hparams, f, indent=4, sort_keys=True)


def main(unused_argv):
    def input_fn(filenames, num_epochs=None, shuffle=True, batch_size=200):
        train_dataset = tf.data.TFRecordDataset(filenames)
        train_dataset = train_dataset.map(parser)
        if shuffle:
            train_dataset = train_dataset.shuffle(buffer_size=batch_size * 10)
        train_dataset = train_dataset.repeat(num_epochs)
        train_dataset = train_dataset.batch(batch_size)
        train_iterator = train_dataset.make_one_shot_iterator()
        features, labels = train_iterator.get_next()
        return features, labels

    train_input = lambda: input_fn(FLAGS.train_dataset, num_epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size)

    eval_input = lambda: input_fn(FLAGS.eval_dataset, num_epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size,
                                  shuffle=False)

    session_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        intra_op_parallelism_threads=0,
        gpu_options=tf.GPUOptions(force_gpu_compatible=True))

    # distribution = tf.contrib.distribute.MirroredStrategy()
    config = tf.estimator.RunConfig(session_config=session_config)
    config = config.replace(model_dir=FLAGS.model_dir, save_checkpoints_steps=1000)
    # estimator = tf.estimator.Estimator(model_fn=cnn_model_fn, config=config)
    estimator = tf.estimator.Estimator(model_fn=tf.contrib.estimator.replicate_model_fn(cnn_model_fn), config=config)

    train_spec = tf.estimator.TrainSpec(train_input, max_steps=100000000)

    eval_spec = tf.estimator.EvalSpec(eval_input, steps=1000)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    tf.logging.info('Saving hyperparameters ...')
    save_hp_to_json()


if __name__ == '__main__':
    tf.app.run()
