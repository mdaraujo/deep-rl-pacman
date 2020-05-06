import numpy as np
import tensorflow as tf

from stable_baselines.a2c.utils import conv, linear, conv_to_fc


def pacman_cnn(scaled_images, **kwargs):
    """
    Custom CNN for Pacman.

    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :return: (TensorFlow Tensor) The CNN output layer
    """

    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=64, filter_size=2, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=128, filter_size=3, stride=2, init_scale=np.sqrt(2), **kwargs))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=128, filter_size=5, stride=3, init_scale=np.sqrt(2), **kwargs))
    layer_3 = conv_to_fc(layer_3)
    return activ(linear(layer_3, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))
