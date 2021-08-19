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


""" model = pickle.load(open("/Users/rdcuy/Documents/VANDY_OLD/templates/weather_aqi.pkl", "wb")) """

@app.route('/')
def index():    
    return render_template('index.html')

@app.route("/scraper/")
def scraping():
    weatherdata = ScrapingWeather.scrape()
    model = tf.keras.models.load_model("weather_aqi")
    prediction = model.predict(weatherdata)
    #print(prediction)
    return render_template('index.html', prediction=prediction)
    #return prediction
    """  return redirect('/', code = 302) """
    
if __name__ == '__main__':
    app.run(host='localhost',debug=True)