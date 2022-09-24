import numpy as np
import tensorflow as tf
import os
from workspace.src.utils import image_preprocessing

model = tf.keras.models.load_model('model')

def verify(model, detection_threshold, verification_threshold):
    results = []
    for image in os.listdir(os.path.join('application_data', 'verification_images')):
        input_img = image_preprocessing(os.path.join('application_data', 'input_image', 'input_image2.jpg'))
        validation_img = image_preprocessing(os.path.join('application_data', 'verification_images', image))
        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)
    
    detection = np.sum(np.array(results) > detection_threshold)

    verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images'))) 
    verified = verification > verification_threshold
    
    return results, verified, verification

results, verified, verification = verify(model, 0.9, 0.8)
print(verified)