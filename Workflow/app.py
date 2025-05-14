
from PIL import Image
from flask import Flask,request ,render_template
import cv2
import numpy as np
from predict import predicter



app = Flask(__name__,template_folder="templates")

template = "test_temp.html"

@app.route('/')

def home():
    return render_template(template)

@app.route('/predict', methods=['POST'])
def predict():
    # Check if request contains file
    if 'imagefile' not in request.files:
        return "No file part in the request"
    
    # Get the file from request
    imagefile = request.files['imagefile']
    image = cv2.imdecode(np.frombuffer(imagefile.read(), np.uint8), -1)
    
    predicted_class = predicter(image)
    
    
    # Return the predicted class as a string
    return render_template(template, prediction_text=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)