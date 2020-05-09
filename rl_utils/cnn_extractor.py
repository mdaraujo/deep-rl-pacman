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
    layer = activ(conv(scaled_images, 'c1', n_filters=64, filter_size=3, stride=1, **kwargs))
    layer = activ(conv(layer, 'c2', n_filters=128, filter_size=5, stride=2, **kwargs))
    layer = activ(conv(layer, 'c3', n_filters=128, filter_size=5, stride=2, **kwargs))
    layer = conv_to_fc(layer)
    return activ(linear(layer, 'fc1', n_hidden=512))
