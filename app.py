from flask import (Flask, render_template, redirect, request, jsonify)
import pandas as pd
import numpy as np
import joblib
import datetime as dt
import pickle
import ScrapingWeather
import json
import html
import tensorflow as tf


app = Flask(__name__)
model = tf.keras.models.load_model("weather_aqi.pkl", "rb")

""" model = pickle.load(open("/Users/rdcuy/Documents/VANDY_OLD/templates/weather_aqi.pkl", "wb")) """

@app.route('/')
def index():    
    return render_template('index.html')

@app.route("/scrape")
def scrape():
    weather_data = ScrapingWeather.scrape()
    prediction = model.predict([np.array(list(weather_data.values()))])
    output = prediction[0]
    return jsonify(output)
    """  return redirect('/', code = 302) """
    
if __name__ == '__main__':
    app.run(host='localhost',debug=True)