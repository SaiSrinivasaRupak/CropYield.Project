import joblib
import numpy as np
from flask import Flask, render_template, request, redirect
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('Home.html')

@app.route('/Predict')
def prediction():
    return render_template('Index.html')

@app.route('/form', methods=["POST"])
def brain():
    Crop_Year = (request.form['Crop_Year'])
    Temperature = float(request.form['Temperature'])
    Humidity = float(request.form['Humidity'])
    Soil_Moisture = float(request.form['Soil_Moisture'])
    Area = float(request.form['Area'])
    Crop_num = request.form['Crop']

        # Creating a list of input values
    values = [[Crop_Year, Temperature, Humidity, Soil_Moisture, Area,Crop_num]]
    
    if Humidity>20 and Humidity<=60 and Soil_Moisture>20 and Soil_Moisture<=70:
        joblib.load('cropmod','r')
        model = joblib.load(open('cropmod','rb'))
        arr = [values]
        arr_reshaped = np.array(arr).reshape(1, -1)  # Reshape the input array
        acc = model.predict(arr_reshaped)
        return render_template('Prediction.html', Prediction=str(acc))
    else:
        return "Sorry...  Error in entered values in the form Please check the values and fill it again"

if __name__ == '__main__':
    app.run(debug=True)











