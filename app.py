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


@app.route('/')
def index():    
    return render_template('index.html')

@app.route("/scraper")
def scraping():
    weatherdata = ScrapingWeather.scrape()
    return render_template('index.html', weatherdata = weatherdata.to_html(classes = 'table table-striped'))

if __name__ == '__main__':
    app.run(host='localhost',debug=True)