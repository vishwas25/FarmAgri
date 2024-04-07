from flask import Flask, render_template, request
from datetime import datetime
import numpy as np
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from keras.models import load_model

# Load models
model = load_model(r"D:\Group ID 45\Farmagri\Farmagri\farmagri\env\model\cotton.h5")
print('@@ Cotton Model loaded')
model1 = load_model(r"D:\Group ID 45\Farmagri\Farmagri\farmagri\env\model\potato.h5")
print('@@ Potato Model loaded')

# Function to predict potato disease
def pred_pot_dieas(pot_plant):
    test_image = load_img(pot_plant, target_size=(256, 256))  # Load image
    print("@@ Got Image for prediction")
    test_image = img_to_array(test_image) / 255  # Convert image to np array and normalize
    test_image = np.expand_dims(test_image, axis=0)  # Change dimension 3D to 4D
    result = model1.predict(test_image).round(3)  # Predict diseased plant or not
    print('@@ Raw result = ', result)
    pred = np.argmax(result)  # Get the index of max value
    if pred == 0:
        return "Early blight", 'potato.html'  # If index 0 burned leaf
    elif pred == 1:
        return 'Late blight', 'potato.html'  # If index 1
    else:
        return "Healthy Cotton Plant", 'potato.html'  # If index 3


# Function to predict cotton disease
def pred_cot_dieas(cott_plant):
    test_image = load_img(cott_plant, target_size=(150, 150))  # Load image
    print("@@ Got Image for prediction")
    test_image = img_to_array(test_image) / 255  # Convert image to np array and normalize
    test_image = np.expand_dims(test_image, axis=0)  # Change dimension 3D to 4D
    result = model.predict(test_image).round(3)  # Predict diseased plant or not
    print('@@ Raw result = ', result)
    pred = np.argmax(result)  # Get the index of max value
    if pred == 0:
        return "bacterial blight", 'cotton.html'  # If index 0 burned leaf
    elif pred == 1:
        return 'curl virus', 'cotton.html'  # If index 1
    elif pred == 2:
        return 'fussarium wilt', 'cotton.html'  # If index 2 fresh leaf
    else:
        return "Healthy Cotton Plant", 'cotton.html'  # If index 3


app = Flask(__name__)

# Route for the HTML form
@app.route('/')
def index(id, name):
    return render_template('index.html')

# Route for form submission
@app.route('/submit', methods=['POST'])
def submit():
    name = request.form['name']
    plant = request.form['plant']
    id = request.form['Lid']
    image = request.files['image']
    image_path = 'static/user_uploaded/' + image.filename
    image.save(image_path)

    print("@@ Predicting class......")
    if plant == 'Cotton':
        pred, output_page = pred_cot_dieas(cott_plant=image_path)
    else:
        pred, output_page = pred_pot_dieas(pot_plant=image_path)

    img = 'user_uploaded/' + image.filename
    return render_template(output_page, pred_output=pred, user_image=img, id=id, name=name)

# Route for success page
@app.route('/success')
def success():
    return 'Data submitted successfully!'

if __name__ == '__main__':
    app.run(debug=True)
