import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image


# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import os
import keras

from tensorflow import keras
from tensorflow.keras import applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import scipy

# ============================================================================
from werkzeug.wrappers import Request, Response
from flask import Flask
# ============================================================================

class_list=['COVID',
    'Normal',
    'Lung_Opacity',
    'Viral Pneumonia']

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'MobileNetCAM.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model.make_predict_function()        # Necessary

def model_predict(img, model):
    print('In model_predict Funct')

    # Preprocessing the image
    x = np.array(img)
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    label_index = np.argmax(preds)
    print(label_index)
    label_class_name = class_list[label_index]
    return label_class_name

#---------------------------------------------------------------------------------------------


def get_class_activation_map(model, img):

    ''' 
    this function computes the class activation map
    
    Inputs:
        1) model (tensorflow model) : trained model
        2) img (numpy array of shape (224, 224, 3)) : input image
    '''

    # expand dimension to fit the image to a network accepted input size
    img = np.expand_dims(img, axis=0)

    # predict to get the winning class
    predictions = model.predict(img)
    label_index = np.argmax(predictions)

    # Get the 2048 input weights to the softmax of the winning class.
    class_weights = model.layers[-1].get_weights()[0]
    class_weights_winner = class_weights[:, label_index]
    
    # get the final conv layer   
    final_conv_layer = model.get_layer("dense")
    
    # create a function to fetch the final conv layer output maps (should be shape (1, 7, 7, 2048)) 
    get_output = K.function([model.input],[final_conv_layer.output, model.layers[-1].output])
    [conv_outputs, predictions] = get_output([img])
 
    # squeeze conv map to shape image to size (7, 7, 2048)
    conv_outputs = np.squeeze(conv_outputs)
    
    # bilinear upsampling to resize each filtered image to size of original image 
    mat_for_mult = scipy.ndimage.zoom(conv_outputs, (32, 32, 1), order=1) # dim: 224 x 224 x 2048
    
    # get class activation map for object class that is predicted to be in the image
    final_output = np.dot(mat_for_mult.reshape((224*224), 512), class_weights_winner).reshape(224,224) # dim: 224 x 224
    

    # return class activation map
    return final_output, label_index

def plot_class_activation_map(CAM, img, label):
    import cv2
    import pathlib
    import os
    ''' 
    this function plots the activation map 
    
    Inputs:
        1) CAM (numpy array of shape (224, 224)) : class activation map containing the trained heat map
        2) img (numpy array of shape (224, 224, 3)) : input image
        3) label (uint8) : index of the winning class
    '''
    
    # plot image
    plt.imshow(img, alpha=0.5)
    
    # plot class activation map
    plt.imshow(CAM, cmap='jet', alpha=0.5)

    # get string for classified class

    class_label = class_list[label]
    plt.title(class_label)
    plt.show()

# --------------------------------------------------------------------------------------------

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    print('In Upload Funct')
    
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        img = keras.utils.load_img(file_path, target_size=(224, 224))
        
        # Make prediction
        preds = model_predict(img, model)
        
        # Plot applied CAM         
        CAM, label = get_class_activation_map(model, img)
        plot_class_activation_map(CAM, img, label) 
        
        return preds
    return None


if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('localhost', 9000, app)
    
    


