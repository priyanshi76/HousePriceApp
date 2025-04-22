from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    area = float(request.form['area'])
    bedrooms = float(request.form['bedrooms'])
    input_features = np.array([[area, bedrooms]])
    prediction = model.predict(input_features)
    return render_template('index.html', prediction_text=f'Estimated Price: ${prediction[0]:,.2f}')

if __name__ == "__main__":
    app.run(debug=True)
