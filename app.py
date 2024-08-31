from flask import Flask, render_template, request, redirect, url_for
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load your Keras model
model = load_model("new_knife_detection_model.h5")
model.make_predict_function()  # Necessary for multi-threaded applications like Flask

# Function to predict image
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(240, 240))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalization
    prediction = model.predict(img_array)
    return prediction

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Prediction page
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        # If no file selected
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            
            # Get prediction
            prediction = predict_image(file_path)
            predicted_class = np.argmax(prediction)
            
            # Render result page with prediction
            return render_template('result.html', prediction=predicted_class)
        
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
