#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import Dependencies for Machine Learning

import joblib
import pickle
from sqlalchemy import create_engine, func
from sqlalchemy.orm import Session
from sqlalchemy.ext.automap import automap_base
import sqlalchemy
import sqlite3 as sq
from imblearn.over_sampling import RandomOverSampler
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
import pandas as pd
import tensorflow as tf
import os
from tensorflow.keras.callbacks import ModelCheckpoint

warnings.filterwarnings('ignore')


# In[2]:


#import dataset
url = "https://drive.google.com/file/d/1uG3yPNRihgE3j2bGQGOAq8iNgMjXlybw/view?usp=sharing"
path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
weather_df = pd.read_csv(path, index_col=0)


# In[3]:


weather_df.head(10)


# In[4]:


#import aqi dataset
url = "https://drive.google.com/file/d/1pvPyRj--mbnipGaNnXDH_Fh7ncFvVZWO/view?usp=sharing"
path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
air_df = pd.read_csv(path, index_col=0)


# In[5]:


air_df.head(10)


# In[6]:


weather_df.columns


# In[7]:


weather_df_clean = weather_df.drop(columns=["State", "Country",
                         "Maximum Temperature", "Minimum Temperature", "Wind Chill", "Snow Depth", "Wind Gust",
                         "Heat Index", "Wind Direction", "Snow", "Visibility", "Cloud Cover" ])
weather_df_clean.head(10)


# In[8]:


weather_df_clean = weather_df_clean.rename(columns={"Date time": "date_time", "Wind Speed": "wind_speed",
                                                   "Relative Humidity": "relative_humidity"})
weather_df_clean.columns


# In[9]:


air_df.columns


# In[10]:


air_df_clean  = air_df.drop(columns=['state', 'county','parameter', 'sample_duration','units_of_measure', 'arithmetic_mean',])
air_df_clean.head(10)


# In[11]:


air_df_clean = air_df_clean.rename(columns={"date_local": "date_time"})
weather_df_clean.columns


# In[12]:


sql_data = 'pollution.sqlite'


# Create connection & push the data:

conn = sq.connect(sql_data)
cur = conn.cursor()

cur.executescript('''
DROP TABLE IF EXISTS "WEATHER";
CREATE TABLE "WEATHER" (
	"index" INTEGER PRIMARY KEY AUTOINCREMENT,
	"city" TEXT NOT NULL,
	"date_time" TEXT NOT NULL,
	"Temperature" INTEGER NOT NULL,
	"Precipitation" TEXT NOT NULL,
	"wind_speed" TEXT NOT NULL,
    "relative_humidity" TEXT NOT NULL,
    "Conditions" TEXT NOT NULL
);
DROP TABLE IF EXISTS "POLLUTION";
CREATE TABLE "POLLUTION" (
	"index" INTEGER PRIMARY KEY AUTOINCREMENT,
	"city" TEXT NOT NULL,
	"date_time" INTEGER NOT NULL,
	"AQI" TEXT NOT NULL
);
''')
# conn.commit()
weather_df_clean.to_sql("WEATHER", conn, if_exists='append', index=True)
# conn.commit()
air_df_clean.to_sql("POLLUTION", conn, if_exists='append', index=True)

conn.commit()
conn.close()


# In[13]:


# Reflect the Tables into SQLAlchemy ORM
engine = create_engine("sqlite:///pollution.sqlite")

# Reflect an existing database into a new model:
Base = automap_base()

# Reflect the tables:
Base.prepare(engine, reflect=True)


Base.classes.keys()


# In[14]:


pollution = Base.classes.POLLUTION
weather = Base.classes.WEATHER
con = sq.connect("pollution.sqlite")

combined_df = pd.read_sql_query("SELECT weather.city, weather.date_time, weather.Temperature,weather.Precipitation, weather.wind_speed, weather.relative_humidity, weather.Conditions,pollution.city, pollution.date_time, pollution.AQI FROM weather INNER JOIN POLLUTION ON weather.date_time = pollution.date_time WHERE weather.city = pollution.city", con)


combined_df


# In[15]:


combined_df2 = combined_df.drop(columns=["city", "date_time"])

combined_df2


# In[16]:


combined_df2["AQI"] = combined_df2.AQI.astype(float)
combined_df2["wind_speed"] = combined_df2.wind_speed.astype(float)
combined_df2["relative_humidity"] = combined_df2.relative_humidity.astype(float)
combined_df2["Precipitation"] = combined_df2.Precipitation.astype(float)
combined_df2.dtypes


# In[17]:


# Binning aqi values
aqi_bins = [-1, 50, 100, 300, 500]
group_names = ['Good', 'Moderate', 'Unhealthy', 'Hazardous']

# Categorize aqi based on the bins.
combined_df2["AQI_Range"] = pd.cut(combined_df2["AQI"], aqi_bins, labels=group_names)

combined_df2


# In[18]:


# Converting text to code value for AQI_Range
combined_df2["AQI_Range"] = combined_df2["AQI_Range"].astype('category').cat.codes
combined_df2 = combined_df2.drop(columns=["AQI"])
combined_df2


# In[19]:


application_cat = combined_df2.dtypes[combined_df2.dtypes == "object"].index.tolist()
application_cat


# In[20]:


# Create a OneHotEncoder instance
enc = OneHotEncoder(sparse=False)

# Fit and transform the OneHotEncoder using the categorical variable list
encode_df = pd.DataFrame(enc.fit_transform(combined_df2[application_cat]))

# Add the encoded variable names to the dataframe
encode_df.columns = enc.get_feature_names(application_cat)
encode_df.head()


# In[21]:


# Merge one-hot encoded features and drop the original
combined_df2 = combined_df2.merge(encode_df, left_index=True,right_index=True)
combined_df2 = combined_df2.drop(columns = application_cat)
combined_df2.head()


# In[22]:


# Split our preprocessed data into our features and target arrays
y = combined_df2["AQI_Range"].values
X = combined_df2.drop("AQI_Range", 1).values

# Split the preprocessed data into a training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


# In[23]:


print('Train', X_train.shape, y_train.shape)
print('Test', X_test.shape, y_test.shape)


# In[24]:


# Create a StandardScaler instances
scaler = StandardScaler()

# Fit the StandardScaler
X_scaler = scaler.fit(X_train)

# Scale the data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)


# In[25]:


# Define the model - deep neural net, i.e., the number of input features and hidden nodes for each layer.
number_input_features = len(X_train[0])
hidden_nodes_layer1 =  75
hidden_nodes_layer2 = 50
hidden_nodes_layer3 = 25

nn = tf.keras.models.Sequential()

# First hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation="relu"))

# Second hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation="relu"))

# Second hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer3, activation="relu"))

# Output layer
nn.add(tf.keras.layers.Dense(units=1, activation="tanh"))

# Check the structure of the model
nn.summary()


# In[26]:


# Compile the model
nn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])


# In[27]:


# Train the model
fit_model = nn.fit(X_train_scaled,y_train,epochs=100)


# In[28]:


# Evaluate the model using the test data
model_loss, model_accuracy = nn.evaluate(X_test_scaled,y_test,verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

nn.save("weather_aqi.pkl")

load_model = tf.keras.models.load_model("weather_aqi.pkl")

load_model.summary()