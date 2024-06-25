import joblib
import pickle
from tensorflow import keras
from flask import Flask, request, jsonify
from keras.models import model_from_json
import pandas as pd
import numpy as np
from joblib import load
import cv2
from skimage.transform import resize



app = Flask(__name__)

with open('./models/model_architecture.pkl', 'rb') as f:
    print("Loading Model Architecture ...")
    model_architecture = pickle.load(f)
    print("Loaded Model Architecture Succesfully !!!")

# Load the model weights from joblib
model_weights = joblib.load('./models/cnn_weights.joblib')
model3=model_from_json(model_architecture)
model3.set_weights(model_weights)


# model = model_from_json(model_architecture)
# model.set_weights(model_weights)
# def img_id(imgg):
#     img=cv2.imread('imgg')
#     img1=resize(img,(64,64))
#     img2=np.expand_dims(img1,axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    print(model_weights)
    print(model_architecture)
  
    file=request.files['file']
    img_data=file.read()
    numpy_array=np.frombuffer(img_data,np.uint8)
    img=cv2.imdecode(numpy_array,cv2.IMREAD_COLOR)
    img1=resize(img,(64,64))
    img2=np.expand_dims(img1,axis=0)
    print(img2.shape)
    result=model3.predict(img2)
    result_string=""
    if result >= 0.5:
        result_string="HAPPY"
    else:
        result_string="SAD"

    return jsonify({'message':f'Image received and processed successfully,{result_string}'})
    



    

if __name__ == '__main__':
    app.run(debug=True)