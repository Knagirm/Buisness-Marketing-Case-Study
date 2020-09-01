import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('modelDT.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    x = int_features.pop(5)
    if x == 1:
        int_features.extend([1,0,0])
    elif x == 2:
        int_features.extend([0,1,0])
    elif x == 3:
        int_features.extend([0,0,1])
    final_features = pd.DataFrame([int_features])
    prediction = model.predict(final_features)
    if prediction[0] == 1:
        return render_template('yes.html')
    else:
        return render_template('no.html')

if __name__ == "__main__":
    app.run(debug=True)