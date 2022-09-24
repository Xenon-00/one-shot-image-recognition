import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Layer, Conv2D, MaxPooling2D, Flatten, Input

def _embedding_layer():
    _input = Input(shape=(105, 105, 3), name='input_image')
    conv1 = Conv2D(64, (10,10), activation='relu')(_input)
    max1 = MaxPooling2D(64, (2,2), padding='same')(conv1)
    conv2 = Conv2D(128, (7,7), activation='relu')(max1)
    max2 = MaxPooling2D(64, (2,2), padding='same')(conv2)
    conv3 = Conv2D(128, (4,4), activation='relu')(max2)
    max3 = MaxPooling2D(64, (2,2), padding='same')(conv3)
    conv4 = Conv2D(256, (4,4), activation='relu')(max3)
    flat = Flatten()(conv4)
    dense = Dense(4096, activation='sigmoid')(flat)
    return Model(inputs=[_input], outputs=[dense], name='embedding')

class L1layer(Layer):
    def __init__(self) -> None:
        super().__init__()
    def call(self, input_embedding, val_embedding):
        return tf.math.abs(input_embedding - val_embedding)

def one_shot_model():
    input_image = Input(shape=(105, 105, 3), name="input_img")
    validation_image = Input(shape=(105, 105, 3), name="validation_img")
    embedding = _embedding_layer()
    l1layer = L1layer()
    l1layer._name = "distance_layer"
    distances = l1layer(embedding(input_image), embedding(validation_image))
    classifier = Dense(1, activation='sigmoid')(distances)
    return Model(inputs=[input_image, validation_image], outputs=[classifier], name='SiameseNetwork')

