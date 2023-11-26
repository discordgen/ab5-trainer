from keras.models import load_model
from flask import Flask, request
from PIL import Image
import numpy as np
from keras.preprocessing.image import img_to_array
import io
import base64
import time
import os

np.set_printoptions(suppress=True)
model = load_model('model.keras')
app = Flask(__name__)

class_names = ['animal with four legs', 'other']


@app.route('/api/predictimages', methods=['POST'])
def predictimages():
    result = []
    data: dict = request.get_json()
    topic = data.get("topic")
    images = data.get("images")
    if topic is None:
        return {"error": "Missing required topic parameter"}, 400
        
    if images is None:
        return {"error": "Missing required images parameter"}, 400


    for image in images:
        image = predict(image)
        result.append(image == topic)
    

    return {'success': True, "result": result}



def predict(b64image):
    imageb = base64.b64decode(b64image) 
    img = img_to_array(Image.open(io.BytesIO(imageb)))
    img = np.expand_dims(img, axis=0)

    try:
        prediction = model.predict(img, verbose=0)
        return class_names[np.argmax(prediction, axis=1)[0]]
    except Exception as e:
        return "error"




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
