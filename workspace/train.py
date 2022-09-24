import os
import tensorflow as tf
import matplotlib.pyplot as plt
from src.utils import image_preprocessing, input_validation_processing
from architecture import one_shot_model
from tensorflow.keras.metrics import Precision , Recall

positive_path = os.path.abspath(os.path.join('data', 'positive'))
negative_path = os.path.abspath(os.path.join('data', 'negative'))
anchor_path = os.path.abspath(os.path.join('data', 'anchor'))

positives = tf.data.Dataset.list_files(positive_path + "\*.jpg").take(300)
negatives = tf.data.Dataset.list_files(negative_path + "\*.jpg").take(300)
anchors = tf.data.Dataset.list_files(anchor_path + "\*.jpg").take(300)

positives_image = tf.data.Dataset.zip((anchors, positives, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchors)))))
negatives_image = tf.data.Dataset.zip((anchors, negatives, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchors)))))
data = positives_image.concatenate(negatives_image)

data = data.map(input_validation_processing)
data = data.shuffle(buffer_size=1000)

train_data = data.take(round(len(data) * .7))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)
val_data = data.skip(round(len(data) * .7)).take(round(len(data) * .3))
val_data = val_data.batch(16)
val_data = val_data.prefetch(8)


model = one_shot_model()
criterion = tf.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam(0.0001)
recall = Recall()
precision = Precision()
checkpoint_dir = os.path.abspath(os.path.join('checkpoints'))
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

def train_step(batch):
    with tf.GradientTape() as tape:
        X = batch[:2]
        y = batch[2]
        yhat = model(inputs=X, training=True)
        loss = criterion(y, yhat)      
    grad = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grad, model.trainable_variables))
    recall.update_state(y, yhat)
    precision.update_state(y, yhat)

    print(f"\tLoss : {loss} ; Recall : {recall.result().numpy()} ; Precision : {precision.result().numpy()}")

def train(data, epochs):
    for epoch in range(1, epochs+1):
        print("\n Epoch {}/{}".format(epoch, epochs))
        progbar = tf.keras.utils.Progbar(len(data), width=50)
        for idx, batch in enumerate(data):
            train_step(batch)
            progbar.update(idx+1)

        if epoch % 10 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)


train(train_data, 50)
model.save('model')




    