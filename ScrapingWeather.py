
# Import Splinter, BeautifulSoup, and Pandas
from splinter import Browser
from bs4 import BeautifulSoup as soup
import pandas as pd
from webdriver_manager.chrome import ChromeDriverManager
from sklearn.preprocessing import StandardScaler,OneHotEncoder

# Set the executable path and initialize Splinter
executable_path = {'executable_path': ChromeDriverManager().install()}
browser = Browser('chrome', **executable_path, headless=False)

# Visit the Weather.com site
url = 'https://weather.com/weather/tenday/l/c497a8fe783a21075e4be0fe8e3851415b88cb2e30a6fa184550e22a7ae728c6'
browser.visit(url)

# Optional delay for loading the page
browser.is_element_present_by_css('div.list_text', wait_time=1)


# Convert the browser html to a soup object and then quit the browser
html = browser.html
weather_soup = soup(html, 'html.parser')

# Use the parent element to find all high temperatures and save as `HighTemps`
HighTemps = weather_soup.find_all('span', class_='DetailsSummary--highTempValue--3Oteu')

#For Loop to grab all high temperatures
High_Temps = []
for temp in HighTemps:
    degrees = temp.get_text()
    High_Temps.append(degrees)
    
# Use the parent element to find all low temperatures and save as `LowTemps`
LowTemps = weather_soup.find_all('span', class_='DetailsSummary--lowTempValue--3H-7I')

#For Loop to grab all low temperatures
Low_Temps = []
for temp in LowTemps:
    degrees = temp.get_text()
    Low_Temps.append(degrees)


# Use the parent element to find all conditions and save as `Conditions`
Conditions = weather_soup.find_all('span', class_='DetailsSummary--extendedData--365A_')

#For Loop to grab all conditions
Conditions_List = []
for condition in Conditions:
    description = condition.get_text()
    Conditions_List.append(description)


# Use the parent element to find all wind speeds and save as `Winds`
Winds = weather_soup.find_all('span', class_='Wind--windWrapper--3aqXJ undefined')

#For Loop to grab all winds
Winds_List = []
for wind in Winds:
    description = wind.get_text()
    Winds_List.append(description)
    

#Cleaning up winds to only have speed
#Should use for loop but easy enough with only fifteen speeds
wind_info = []
wind_speeds = []
for w in Winds_List:
    info = w.split()
    wind_info.append(info)
    
wind_speeds.append(wind_info[0][1])
wind_speeds.append(wind_info[1][1])
wind_speeds.append(wind_info[2][1])
wind_speeds.append(wind_info[3][1])
wind_speeds.append(wind_info[4][1])
wind_speeds.append(wind_info[5][1])
wind_speeds.append(wind_info[6][1])
wind_speeds.append(wind_info[7][1])
wind_speeds.append(wind_info[8][1])
wind_speeds.append(wind_info[9][1])
wind_speeds.append(wind_info[10][1])
wind_speeds.append(wind_info[11][1])
wind_speeds.append(wind_info[12][1])
wind_speeds.append(wind_info[13][1])
wind_speeds.append(wind_info[14][1])

# Use the parent element to find all chances of rain and save as `PrecipChances`
PrecipChances = weather_soup.find_all('span', class_='DailyContent--value--37sk2')

#For Loop to grab all precips
Precips_List = []
for precip in PrecipChances:
    chance = precip.get_text()
    Precips_List.append(chance)

Precips_List_Final = [(Precips_List[0]), (Precips_List[4]), (Precips_List[8]), (Precips_List[12]), (Precips_List[16]),
                     (Precips_List[20]), (Precips_List[24]), (Precips_List[28]), (Precips_List[32]), (Precips_List[36]),
                     (Precips_List[40]), (Precips_List[44]), (Precips_List[48]), (Precips_List[52]), (Precips_List[56])]


# Use the parent element to find all humidity readings and save as `Humidities`
Humidities = weather_soup.find_all('span', class_='DetailsTable--value--1q_qD')


#For Loop to grab all humidities
Hums_List = []
for hum in Humidities:
    percent = hum.get_text()
    Hums_List.append(percent)

Hums_List_Final = [(Hums_List[0]), (Hums_List[8]), (Hums_List[16]), (Hums_List[24]), (Hums_List[32]),
                     (Hums_List[40]), (Hums_List[48]), (Hums_List[56]), (Hums_List[64]), (Hums_List[72]),
                     (Hums_List[80]), (Hums_List[88]), (Hums_List[96]), (Hums_List[104]), (Hums_List[112])]


# 5. Quit the browser
browser.quit()


#Creating Forecast Dataframe
forecast_df = pd.DataFrame()

forecast_df['HighTemps'] = High_Temps
forecast_df['LowTemps'] = Low_Temps
forecast_df['Conditions'] = Conditions_List
forecast_df['Winds'] = wind_speeds
forecast_df['Precipitation'] = Precips_List_Final
forecast_df['Humidity'] = Hums_List_Final


#Changing high temp of "--" to an average because when ran at night, get "-- instead of the high temp
#First converting to string
forecast_df['HighTemps'] = forecast_df['HighTemps'].astype('string')
#Converting
forecast_df['HighTemps'] = forecast_df['HighTemps'].str.replace("--", "85", case = False)

#Removing symbols
forecast_df['HighTemps'] = forecast_df['HighTemps'].str.rstrip("°")
forecast_df['LowTemps'] = forecast_df['LowTemps'].str.rstrip("°")
forecast_df['Precipitation'] = forecast_df['Precipitation'].str.rstrip("%")
forecast_df['Humidity'] = forecast_df['Humidity'].str.rstrip("%")


#Converting to out of strings
forecast_df['HighTemps'] = forecast_df['HighTemps'].astype(float)
forecast_df['LowTemps'] = forecast_df['LowTemps'].astype(float)
forecast_df['Winds'] = forecast_df['Winds'].astype(float)
forecast_df['Precipitation'] = forecast_df['Precipitation'].astype(float)
forecast_df['Humidity'] = forecast_df['Humidity'].astype(float)

#Converting Precipitation to decimal
forecast_df['Precipitation'] = (forecast_df['Precipitation'] / 100)


#Transforming Conditions text to match with model
forecast_df['Conditions'] = forecast_df['Conditions'].str.replace("Partly Cloudy", "Partially cloudy", case = False)
forecast_df['Conditions'] = forecast_df['Conditions'].str.replace("Thunderstorms", "Rain", case = False)
forecast_df['Conditions'] = forecast_df['Conditions'].str.replace("Scattered ", "", case = False)
forecast_df['Conditions'] = forecast_df['Conditions'].str.replace("AM Showers", "Rain", case = False)
forecast_df['Conditions'] = forecast_df['Conditions'].str.replace("AM ", "", case = False)
forecast_df['Conditions'] = forecast_df['Conditions'].str.replace("PM ", "", case = False)
forecast_df['Conditions'] = forecast_df['Conditions'].str.replace("Heavy ", "", case = False)
forecast_df['Conditions'] = forecast_df['Conditions'].str.replace("PM Rain", "Rain", case = False)
forecast_df['Conditions'] = forecast_df['Conditions'].str.replace("Isolated ", "", case = False)
forecast_df['Conditions'] = forecast_df['Conditions'].str.replace("Mostly Sunny", "Clear", case = False)


# Generate our categorical variable lists
application_cat = forecast_df.dtypes[forecast_df.dtypes == "object"].index.tolist()


# Create a OneHotEncoder instance
enc = OneHotEncoder(sparse=False)

# Fit and transform the OneHotEncoder using the categorical variable list
encode_df = pd.DataFrame(enc.fit_transform(forecast_df[application_cat]))

# Add the encoded variable names to the dataframe
encode_df.columns = enc.get_feature_names(application_cat)

# Merge one-hot encoded features and drop the originals
forecast_df_final = forecast_df.merge(encode_df, left_index=True,right_index=True)
forecast_df_final = forecast_df_final.drop(columns = application_cat)

# Drop one of the temp columns
forecast_df_final = forecast_df_final[['HighTemps', 'Winds', 'Precipitation', 'Humidity', 'Conditions_Partially cloudy', 'Conditions_Rain']]
forecast_df_final['Temperature'] = forecast_df_final['HighTemps']
#Drop Low Temps column
forecast_df_final = forecast_df_final[['Temperature', 'Winds', 'Precipitation', 'Humidity', 'Conditions_Partially cloudy', 'Conditions_Rain']]


