import tensorflow as tf
import numpy as np

def image_preprocessing(filepath):
    byte_img = tf.io.read_file(filepath)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (105, 105))
    img = img / 255
    return img

def input_validation_processing(input_img, validation, label):
    return (image_preprocessing(input_img), image_preprocessing(validation), label)

