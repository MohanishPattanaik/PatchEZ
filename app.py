from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load("stress_model.pkl")
scaler = joblib.load("scaler.pkl")

# Initialize Flask app
app = Flask(__name__, template_folder='templates')

# Route to render the HTML page and css file

@app.route('/')
def home():
    return "PatchEZ Flask App is Live 🚀"

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/blogs')
def blogs():
    return render_template('blogs.html')

@app.route('/contact')
def contact():
    return render_template('Contacts.html')

@app.route('/styles')
def styles():
    return app.send_static_file("styles.css")

@app.route('/images/Mohanish.png')
def get_image(filename):
    return app.send_static_file(f'images/{filename}')




# API route for predicting stress from single input
@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from the form
    vitals = [
        float(request.form['heart_rate']),
        float(request.form['respiration_rate']),
        float(request.form['snoring_range']),
        float(request.form['body_temperature']),
        float(request.form['limb_movement']),
        float(request.form['eye_movement']),
        float(request.form['hours_of_sleep']),
        float(request.form['blood_oxygen'])
    ]

    # Convert to numpy array and reshape for scaling
    input_data = np.array(vitals).reshape(1, -1)

    # Standardize the input
    scaled_data = scaler.transform(input_data)

    # Predict stress level
    prediction = model.predict(scaled_data)

    return jsonify({"stress_level": int(prediction[0])})



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=10000)
