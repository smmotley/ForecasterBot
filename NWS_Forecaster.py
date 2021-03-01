"""
Data Extraction and Plotting for various Weather APIs
===========================

"""
from noaa_sdk import noaa
import requests
import pathlib
import pandas as pd
from datetime import datetime
from datetime import timedelta
from dateutil import parser
import pytz
import json
from pandas.io.json import json_normalize
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
from matplotlib import cm
import time
import calendar
import numpy as np
import math
import os
import sqlite3
from pathlib import Path
import mailer
from PIL import Image
import platform
from anticaptchaofficial.recaptchav2proxyless import *

plt.interactive(False)
KEYS = pd.read_csv(os.path.join(Path(__file__).parents[2], 'userkeys.config'))
n = noaa.NOAA()
class MplColorHelper:

    def __init__(self, cmap_name, start_val, stop_val):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val):
        return self.scalarMap.to_rgba(val)


def main():
    locations = {'city_code':
                     {'KSAC': {'latlng': (38.61, -121.41), 'longname': 'Sacramento'},
                      'KBUR': {'latlng': (34.23, -118.48), 'longname': 'Burbank'},
                      'KPDX': {'latlng': (45.53, -122.64), 'longname': 'Portland'},
                      'KSEA': {'latlng': (47.65, -122.30), 'longname': 'Seattle'},
                      'KLAS': {'latlng': (36.17, -115.18), 'longname': 'Las Vegas'},
                      'KPHX': {'latlng': (33.45, -112.07), 'longname': 'Phoenix'}
                      }}
    alerts = nwsAlerts(locations)

    historical_date = (datetime.now(pytz.timezone('US/Pacific'))) + timedelta(days=-1)
    historical_date = historical_date.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
    df_historical = historical(list(locations['city_code']), historical_date)
    sql_inject(df_historical, location='historical_data', actuals=True)
    # df_hourly = hourly(n, convert_to_daily=True)
    df_models = stormVista(list(locations['city_code']))

    for location in locations['city_code']:
        df_daily = daily(n, locations['city_code'][location]['latlng'])

        # An empty dataframe will occur if the Aries Weather data pull failed. If the NWS data fails, the
        # df will still have Aries weather info. In this case, the 'nws_high' column will be nan, but
        # the df index will still contain datetimes from Aries, so a merge CAN take place with df_models.
        if df_daily.empty:
            # At this point df_daily is completely empty (no index, no nan's, nothing), so the following line
            # of code simply adds nan values in each row of the dataframe for the nws and aries weather columns.
            # Note: In the final plotting function, the code says: if the nws / aries values are nan, then use a
            #       formula consisting of only model data.
            df_final = df_models.reindex_axis(df_models.columns.union(df_daily.columns), axis=1)
        else:
            df_final = pd.merge(df_models.filter(like=location + '_max', axis='columns'), df_daily,
                                left_on=[df_models.index.month, df_models.index.day],
                                right_on=[df_daily.index.month, df_daily.index.day],
                                how='left').set_index(df_models.index)

        # Extract the entire row within df_historical for the given station
        df_historical_city = df_historical.loc[df_historical['station'] == location]

        # Must return a dataframe since we're manipulating df_final in this def
        #if location != "KLAS":
        df_maxt = plot(df_final, location, df_historical_city,
                       long_name = locations['city_code'][location]['longname'])
        sql_inject(df_maxt, location, actuals=False)

    # Merge all city images into one image.
    create_merged_png()

    # Go to WxBell and get latest image from 00Z model
    wx_bell_img = wxBell()

    img_dir = os.path.join(os.path.sep, 'home','smotley','images','weather_email')

    # On the Linux, save all the files to the img directory.
    this_dir = img_dir
    snowlevel_bot_dir = img_dir
    res_snopack_chart_dir = img_dir
    swe_dir = this_dir

    # On windows, save to the correct folders
    if platform.system() == 'Windows':
        img_dir = pathlib.Path('G:/','Energy Marketing','Weather','Programs','Lake_Spaulding')
        this_dir = os.path.dirname(__file__)
        snowlevel_bot_dir = os.path.join(os.path.dirname(__file__), "..", "SnowLevel_Bot", "Images")
        res_snopack_chart_dir = "U:\Documents\Programming\PycharmProjects\Prod\Reservior_Snowpack_Chart\Images"
        swe_dir = "G:\Energy Marketing\Weather"

    run_mailer = mailer.send_mail(
        alerts,
        os.path.join(this_dir, 'All_Cities.png'),
        os.path.join(this_dir, "Precip_Image", wx_bell_img),
        os.path.join(swe_dir, historical_date.strftime('%Y%m%d')+'_SWE.jpg'),
        os.path.join(snowlevel_bot_dir, "qpf_graph.png"),
        os.path.join(res_snopack_chart_dir, 'MFP_Combined_Storage.png'),
        os.path.join(img_dir, 'LSP_Graphs.png'))
    # df_final.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),'output.csv'), index=True,

def historical(cities,date):
    hist = n.historical_data(date)
    df_hist = pd.concat([pd.DataFrame(json_normalize(x['properties'])) for x in hist['features']], ignore_index=True)
    df_hist = df_hist[df_hist['station'].isin(cities)]

    # Need to convert list objects to strings since SQL can not take a list
    df_hist["high_record_years"] = df_hist["high_record_years"].apply(lambda x: list(map(str, x)), 1).str.join(',')
    df_hist["low_record_years"] = df_hist["low_record_years"].apply(lambda x: list(map(str, x)), 1).str.join(',')
    df_hist["date"] = date
    df_hist.set_index("date", inplace= True)
    return df_hist

def sql_inject(df, location, actuals):
    pd.options.mode.chained_assignment = None  # Turns off a warning that we are copying a dataframe
    db_path = os.path.join(os.path.dirname(__file__), 'forecast_data.sqlite3')
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Need to convert to tz-naive if injecting into sqlite (actuals have already been converted).
    if not actuals:
        df.index = df.index.tz_convert(None)

    # df.columns = [str(col) + '_Tmax' for col in df.columns]
    df['city_code'] = location
    df['date_created'] = df.index[0]
    df['date_valid'] = df.index
    df['forecast_day'] = (df['date_valid'] - df['date_created'])/np.timedelta64(1, 'D')
    if actuals:
        # Remove new parameters in the actuals file that are not in our database
        df.drop(['high_time',
                 'low_time',
                 'precip_normal',
                 'precip_record_years',
                 'snow_record_years', 'average_wind_speed',
                 'highest_gust_direction', 'highest_gust_speed',
                 'highest_wind_direction', 'highest_wind_speed',
                 'resultant_wind_direction', 'resultant_wind_speed',
                 'valid', 'average_sky_cover', 'precip_jan1_depart'],
                inplace=True, axis=1)
        c.execute("SELECT date_created FROM actuals WHERE date_created = ? AND city_code = ?",
                  (df.index[0].strftime('%Y-%m-%d %H:%M:%S'), location,))
        data = c.fetchall()
        if len(data) == 0:
            print("INJECTING HISTORICAL DATA INTO SQL DATABASE...\n")
            df.to_sql('actuals', conn, if_exists='append', index=False)
        else:
            print("HISTORICAL DATA ALREADY EXISTS IN DATABASE")
        return
    c.execute("SELECT date_created FROM forecasts WHERE date_created = ? AND city_code = ?",
              (df.index[0].strftime('%Y-%m-%d %H:%M:%S'), location,))
    data=c.fetchall()
    if len(data)==0:
        print("Appending Data To Forecast Table of Database")
        df.to_sql('forecasts', conn, if_exists='append', index=False)
    else:
        print("Data already exists for " + location + " on " + (df.index[0]).strftime("%Y-%m-%d"))
    pd.options.mode.chained_assignment = 'warn'  # Turn warning back on
    return


def hourly(n, latlng, convert_to_daily):
    # 1) Grab the data from the interwebs
    res_nws = n.points_forecast(latlng[0], latlng[1], hourly=True)     # NWS POINT FORECAST
    res_wu = n.wu_forecast(latlng[0], latlng[1], hourly=True)          # Wunderground POINT FORECAST
    res_aw = n.aw_forecast(latlng[0], latlng[1], hourly=True)          # AerisWeather POINT FORECAST

    # 2) Make dataframe objects from json data
    # The WU forecast must be flattened to create dataframe.
    df_wu_hourly = pd.concat([pd.DataFrame(json_normalize(x)) for x in res_wu['hourly_forecast']], ignore_index=True)
    df_nws_hourly = pd.DataFrame(res_nws['properties']['periods'])

    # 3) Take the string object and convert to a datetime object.
    # Set the column FCTTIME.UTCDATE (which is currently empty) to the 'epoch' object in the json file.
    df_wu_hourly['FCTTIME.UTCDATE'] = [datetime.utcfromtimestamp(float(t['FCTTIME']['epoch'])) for t in res_wu['hourly_forecast']]

    # Make sure the column 'startTime' is a datetime object for Pandas to read
    df_nws_hourly['startTime'] = pd.to_datetime(df_nws_hourly['startTime'])

    # 4) Convert times to timezone aware objects
    # Provide timezone information to our dataframe by first setting the time information to UTC.
    df_wu_hourly['FCTTIME.UTCDATE'] = df_wu_hourly['FCTTIME.UTCDATE'].dt.tz_localize(pytz.utc)

    # Create a dataframe for the NWS forecast, make this timezone aware by first setting it to UTC.
    df_nws_hourly['startTime'] = df_nws_hourly['startTime'].dt.tz_localize(pytz.utc)

    # 5) Now that we have two dataframes, both with datetime columns in UTC,
    #   we can merge them based off the datetime column.
    df_fcst_hourly = pd.merge(df_wu_hourly, df_nws_hourly, left_on='FCTTIME.UTCDATE',right_on='startTime', how='left')
    df_fcst_hourly[['temp.english','dewpoint.english']]=df_fcst_hourly[['temp.english','dewpoint.english']].apply(pd.to_numeric)

    # 6) We now have one dataframe, in UTC. Convert datetime to local time before resampling from hourly to daily.

    # Get timezone information. NOTE, we are using timezone info from NWS only since
    #  the locations of the WU and NWS are the same.
    parsedDate = parser.parse(res_nws['properties']['periods'][0]['startTime'])

    df_fcst_hourly['FCTTIME.UTCDATE'] = df_fcst_hourly['FCTTIME.UTCDATE'].dt.tz_convert(parsedDate.tzinfo)
    df_fcst_hourly.set_index('FCTTIME.UTCDATE',inplace=True)

    df_fcst_hourly.rename(columns = {'temp.english' : 'wu_temp', 'temperature' : 'nws_temp'}, inplace=True)
    # This was just a test to prove that the resampling of hourly data to daily data would keep everything
    # in the correct timezone
    # df_fcst_hourly.at['2018-07-22T21:00:00.000000000','nws_temp'] = 115

    df_fcst_hourly.plot(y=['wu_temp', 'nws_temp'], use_index= True, kind='line')

    #plt.show()

    # 7) Resample hourly data into daily data based off of local timezone information
    df_fcst_daily = df_fcst_hourly.resample('D')['wu_temp','nws_temp'].agg({'min','max'})

    # If you don't joint the columns names, the names will be a tuple which is difficult to access
    df_fcst_daily.columns = ['_'.join(col).strip() for col in df_fcst_daily.columns.values]
    if convert_to_daily:
        return df_fcst_daily
    return df_fcst_hourly


def daily(n, latlng):
    # 1) Get json data from NWS and WU response. We are getting daily data here, so set hourly flag to FALSE
    res_nws = []
    try:
        res_nws = n.points_forecast(latlng[0], latlng[1], hourly=False)  # NWS POINT FORECAST
    except:
        print("Did Not Grab NWS Forecast...Continuing")
    # Turning off weather underground due to end of service on 12/31/18
    #res_wu = n.wu_forecast(latlng[0], latlng[1], hourly=False)  # Wunderground POINT FORECAST
    res_aw = n.aw_forecast(latlng[0], latlng[1], hourly=False)  # AerisWeather POINT FORECAST

    # 2) Make dataframe object from the json response
    # Make a dataframe object of the WU forecast by flattening out the json file.
    # df_wu = pd.concat([pd.DataFrame(json_normalize(x)) for x in res_wu['forecast']['simpleforecast']['forecastday']], ignore_index=True)
    try:
        df_aw = pd.concat([pd.DataFrame(json_normalize(x)) for x in res_aw['response'][0]['periods']],ignore_index=True)
        # 3) Take the string object and convert to a datetime object.
        #   Set the column 'day' to the 'epoch' object in the json file.
        #   Since the NWS must match these dates, we just use WU dates.
        df_aw['day'] = [datetime.utcfromtimestamp(float(t['timestamp'])) for t in res_aw['response'][0]['periods']]

        # 4) Get time zone data and convert any datetime columns to datetime objects that Pandas can read.
        # Get the timezone location for our point from Wunderground's json file.
        # wu_timezone = res_wu['forecast']['simpleforecast']['forecastday'][0]['date']['tz_long']
        aw_timezone = res_aw['response'][0]['profile']['tz']

        # Provide timezone information to our dataframe by first setting the time information to UTC,
        # then converting to the actual timezone
        df_aw['day'] = df_aw['day'].dt.tz_localize(pytz.utc)
        df_aw['day'] = df_aw['day'].dt.tz_convert(aw_timezone)
        df_aw.rename(columns={'maxTempF': 'high_aw', 'minTempF': 'low_aw'}, inplace=True)
        df_aw[['high_aw', 'low_aw']] = df_aw[['high_aw', 'low_aw']].apply(pd.to_numeric)

    except IndexError:
        # If the Aw data doesn't exist, we won't even try to get the NWS data, just return an empty dataframe
        # which will lead to the forecast coming entirely from the model.
        print("PLEASE RESET YOUR AERISWEATHER KEY, RETURNING EMPTY DATAFRAME FOR AW AND NWS")
        df_empty = pd.DataFrame(columns=["high_aw", "high_nws"])
        return df_empty
    try:
        df_nws = pd.DataFrame(res_nws['properties']['periods'])

        # 4) Get time zone data and convert any datetime columns to datetime objects that Pandas can read.
        # Get the timezone location for our point from Wunderground's json file.
        # wu_timezone = res_wu['forecast']['simpleforecast']['forecastday'][0]['date']['tz_long']
        nws_timezone = parser.parse(res_nws['properties']['periods'][0]['startTime'])

        # Make sure the time columns are a datetime object for Pandas to read
        df_nws['startTime'] = pd.to_datetime(df_nws['startTime'], utc=True)
        df_nws['startTime'] = df_nws['startTime'].dt.tz_convert(nws_timezone.tzinfo)
        df_nws['endTime'] = pd.to_datetime(df_nws['endTime'], utc=True)
        df_nws['endTime'] = df_nws['endTime'].dt.tz_convert(nws_timezone.tzinfo)

        # 5) NWS Only: There is no "high" or "low" column in the data, so we have to create one:
        # Instead of combining the data by a single day, the NWS provides a start and end
        # time for each period with an "isDaytime" flag. We will use the isDaytime flag to get
        # the daytime and nighttime temperaures.
        df_nws['high_nws'] = df_nws[df_nws.isDaytime.isin([True])]['temperature']
        df_nws['low_nws'] = df_nws[df_nws.isDaytime.isin([False])]['temperature']

        # We want to merge the dataframes off of a unique date, but the NWS data has the same date
        # muliple times. Therefore, we will just make a "day and night" dataframe that has only one day
        # per entry, which will allow us to merge that data with the wUnderground data.
        df_nws_day = df_nws[pd.notnull(df_nws['high_nws'])]
        df_nws_night = df_nws[pd.notnull(df_nws['low_nws'])]

        # 6) Merge the two dataframes twice: Once to get the NWS high temperatures into the df and once to get the low
        # temperatures. After this is done, we have one dataframe that we can return.
        df_aw = pd.merge(df_aw, df_nws_day[['high_nws']], left_on=[df_aw.day.dt.month, df_aw.day.dt.day],
                         right_on=[df_nws_day['endTime'].dt.month, df_nws_day['endTime'].dt.day], how='left')

        # for some reason, the merge puts in a key_0, key_1 col, which needs to be deleted before we do the next merge.
        df_aw.drop(['key_0', 'key_1'], axis=1, inplace=True)

        df_aw = pd.merge(df_aw, df_nws_night[['low_nws']], left_on=[df_aw.day.dt.month, df_aw.day.dt.day],
                         right_on=[df_nws_night['endTime'].dt.month, df_nws_night['endTime'].dt.day], how='left')

        # for some reason, the merge puts in a key_0, key_1 col, which needs to be deleted before we do the next merge.
        df_aw.drop(['key_0', 'key_1'], axis=1, inplace=True)

        df_aw.set_index('day', inplace=True)
        # Note: even though the df_wu will display in UTC if you view the df, it is still timezone aware and
        #       will contain info for the correct timezone (prove it by uncommenting the print statement below).
        # print(df_wu.index[0].day)

    except:
        print("!!!!FATAL ERROR!!!! NWS Data Unavailable: ")
        df_nws = pd.DataFrame(columns=["endTime", "startTime", "high_nws", "low_nws"])

        # IF you get an exception, at this point the df_aw index will not be set. So set index as date.
        df_aw.set_index('day', inplace=True)
        df_aw = df_aw.reindex_axis(df_aw.columns.union(df_nws.columns), axis=1)
    return df_aw

def nwsAlerts(warning_types):
    # Link to CA zone map: https://www.weather.gov/media/pimar/PubZone/ca_n_zone.pdf
    counter = 0
    msg = ""
    id_list = []
    try:
        alerts = n.alerts(active="1", zone="CAZ067,CAZ017,CAZ069")
        for alert in alerts["features"]:
            # Since we are getting alerts from multiple Zones, it's possible to have the exact same alert for different
            # zones. Therefore, we need to make sure we're not sending multiple alerts for the same alert in the email.
            # ID is in the form of "https://api.weather.gov/alerts/NWS-IDP-PROD-3627045-3151056" So we split it twice
            # The first split gives the ID # and another # (e.g. 3627045-3151056) so we split by the '-' and take the
            # first number, which I think is the actual ID of the warning.
            thisID = alert["id"].split("NWS-IDP-PROD-")[1].split('-')[0]
            if thisID not in id_list:
                id_list.append(thisID)
                counter = counter + 1
                msg = msg + f"""\
                   <strong><br> <span style="color:red;">Alert # {str(counter)} </span><br> {alert["properties"]["headline"]} 
                   </strong> <br> <b>Summary:</b> <br> {alert["properties"]["description"]} <br> """
    except:
        counter = "ERROR. NWS ALERT SERVICE UNAVILABLE."

    htmlMsg = f"""\
    <html>
      <head></head>
      <body>
        <p><h3>There Are <b>{counter}</b> Active NWS Alerts In The PCWA Region:</h3>
           {msg}
           <br>
        </p>
      </body>
    </html>
    """

    return htmlMsg


def stormVista(cities):

    base_url = "https://www.stormvistawxmodels.com/"
    clientKey =  KEYS.iloc[0]['key']
    models = ["gfs", "ecmwf", "gfs-ens-bc", "ecmwf-eps"]
    # hours = np.arange(0,360,6)
    today = datetime.utcnow()
    tomorrow = (datetime.utcnow() + timedelta(days=1)).strftime("%Y%m%d")

    modelCycle = '12z'
    # General rule that the 12Z model is not out until roughly 1:00 pm PDT (20z)
    # and the 00Z isn't avail until 1:00 am (0800Z)
    if 8 <= today.hour <= 20:
        modelCycle = '00z'

    # Create and empty dataframe that will hold dates in the ['date'] column for 16 days.
    df_models = pd.DataFrame(data=pd.date_range(start=today.strftime("%Y%m%d"), periods=16, freq='D'), columns=['date'])

    for model in models:
        # raw = "client-files/" + clientKey + "/model-data/" + model + "/" + today + "/" +
        # modelCycle + "/city-extraction/individual/" + regions + "_raw.csv"
        #CORRECTED:
        #sv_min_max = "client-files/" + clientKey + "/model-data/" + model + "/" + today.strftime("%Y%m%d") + "/" + modelCycle + "/city-extraction/corrected-max-min_northamerica.csv"

        #RAW
        sv_min_max = "client-files/" + clientKey + "/model-data/" + model + "/" + today.strftime(
            "%Y%m%d") + "/" + modelCycle + "/city-extraction/max-min_northamerica.csv"
        fileName = today.strftime("%Y%m%d") + "_" + modelCycle + "_" + model + ".csv"

        curDir = os.path.dirname(os.path.abspath(__file__))
        archive_dir = os.path.join(curDir, "Archive")
        # Download file if it hasn't been downloaded yet.
        if not os.path.isfile(os.path.join(archive_dir,fileName)):
            try:
                df_min_max = pd.read_csv(base_url + sv_min_max, header=[0, 1])
                time.sleep(5)  # Wait 5 seconds before continuing (per stormvista api requirement)
                df_min_max.to_csv(os.path.join(archive_dir,fileName), index=False)
            except:
                return pd.DataFrame(data=pd.date_range(start=today,
                                             end=(datetime.utcnow() + timedelta(days=15)).strftime("%Y%m%d"),
                                                       freq='D'), columns=['date'])
        else:
            df_min_max = pd.read_csv(os.path.join(archive_dir, fileName), header=[0, 1])

        # Because the header info is in the top two rows, pandas treats each column name as a tuple. Flatten the
        # tuple and put a "/" between each string (column 1 = tuple ("A","B") => "A/B"
        df_min_max.columns = df_min_max.columns.map('/'.join)
        df_min_max.set_index(['station/station'], inplace=True)

        #The first 4 charactors in the string will be either "max/" or "min/" followed by the date.
        dates = [c[:-4] for c in df_min_max.columns]
        df = pd.DataFrame(data=pd.date_range(start=dates[0],
                                             end=dates[-1], freq='D'), columns=['date'])

        df_mins = df_min_max[[col for col in df_min_max if 'min' in col]]
        df_maxs = df_min_max[[col for col in df_min_max if 'max' in col]]

        df_mins.columns = [pd.to_datetime(c[:-4]) for c in df_mins.columns]
        df_maxs.columns = [pd.to_datetime(c[:-4]) for c in df_maxs.columns]

        # df_min = pd.DataFrame(data=df_mins.loc[['KSAC', 'KBLU']].T)
        df_min = pd.DataFrame(data=df_mins.loc[cities].T)
        df_min.index.name = 'date'
        df_min.columns = [city + "_min_" + model for city in df_min.columns]

        df_max = pd.DataFrame(data=df_maxs.loc[cities].T)
        df_max.index.name = 'date'
        df_max.columns = [city + "_max_" + model for city in df_max.columns]

        df_models = pd.merge(df_models, df_min, on='date', how='left')
        df_models = pd.merge(df_models, df_max, on='date', how='left')

        # dates = [c for c in df_min_max.columns if c[-2:] != '.1' and c != 'station']
        # df_min_max.set_index(['station_station'], inplace=True)

    getEnsMembers = False
    if getEnsMembers == True:
        dates = list(map(lambda t: (datetime.now() + timedelta(days=t)).strftime("%Y%m%d"), range(15)))
        model = 'ecmwf-eps'
        ens_var = 'tmp2m'
        for date in dates:
            ens_corrected = "client-files/" + clientKey + "/model-data/" + model + "/" + today + "/" + modelCycle + "/city-extraction/d" + today + "_corrected_members_" + ens_var + "_northamerica_06z-06z.csv"
            df_ens = pd.read_csv(base_url+ens_corrected)
    df_models['date'] = df_models['date'].dt.tz_localize(pytz.utc)
    df_models.set_index('date', inplace=True)
    return df_models

def wxBell():
    '''
    WeatherBell has a new map page that sends various post requests, which return the final URL of the image.
    The webhooks are randomized, so we will have to get the randomized names through our own post requests.
    '''

    # For GIF
    get_GIF = False

    # To obtain this, go to https://www.weatherbell.com/login-captcha and search for "sitekey"
    # On their site it's specifically <div class="g-recaptcha" data-sitekey="6LcBa8EZAAAAAICUmvcgJjc0U5KLCPBQN6kNmQ9W">
    wx_bell_captcha_site_key = "6LcBa8EZAAAAAICUmvcgJjc0U5KLCPBQN6kNmQ9W"
    anti_captcha_key = os.environ['ANTI_CAPTCHA_KEY']

    file_date = datetime.now().strftime("%Y%m%d") + "00"
    file_date_06Z = datetime.now().strftime("%Y%m%d") + "06"
    pattern = "%Y%m%d%H"

    # WxBells internal API uses epoch to grab the correct image (instead of 2019-05-01 00Z, it'll do the epoch time)
    epoch = str(int(calendar.timegm(time.strptime(file_date, pattern))))
    epoch_06Z = str(int(calendar.timegm(time.strptime(file_date_06Z, pattern))))

    meteogram_payload = {
        "action": "forecast",
        "type": "meteogram",
        "product": "ecmwf-ensemble",
        "domain": "KBLU",
        "param": "indiv_qpf",
        "init": epoch
    }

    nam_nest_payload = {
        "action": "forecast",
        "domain": "norcal",
        "init": epoch_06Z,
        "param": "refc",
        "product": "nam-nest-conus",
        "type": "model"
    }

    g_response = ""
    try:
        solver = recaptchaV2Proxyless()
        solver.set_verbose(1)
        solver.set_key(anti_captcha_key)
        solver.set_website_url("https://www.weatherbell.com/login-captcha")
        solver.set_website_key(wx_bell_captcha_site_key)

        g_response = solver.solve_and_return_solution()
    except:
        print("COULD NOT SOLVE CAPTCHA")

    req_url = 'https://maps.api.weatherbell.com/image/'  # Base API directory for images.
    req_url_gif = 'https://maps.api.weatherbell.com/gif/'   # Base API directory for gif.
    # old_file_name = 'KBLU_' + file_date + '_eps_precip_360.png'
    # old_image_url = 'http://models.weatherbell.com/ecmwf/' + file_date + '/station/' + old_file_name
    user, pw = KEYS.iloc[3]['short_name'], KEYS.iloc[3]['key']
    payload = {'username': user,
               'password': pw,
               'do_login': 'Login',
               'recaptchaResponse': g_response
               }


    curDir = os.path.dirname(os.path.abspath(__file__))  # Where to put the file
    with requests.Session() as s:
        img_dir = os.path.join(os.path.sep, 'home', 'smotley', 'images', 'weather_email')
        if platform.system() == 'Windows':
            # If we're on windows, just save it to the current folder.
            img_dir = os.path.dirname(os.path.realpath(__file__))

        s.post('https://www.weatherbell.com/login-captcha', data=payload)  # Send post request to login and generate a session.
        meteogram_id = json.loads(s.post(req_url, json=meteogram_payload).text)  # Get the unique ID for the meteogram
        nam_ref_id = json.loads(s.post(req_url, json=nam_nest_payload).text)  # Get the unique ID for the nam nest

        # METEOGRAM SECTION
        # If the file doesn't exist, unique_id[0] will be empty and the url_epoch
        try:
            #uncomment the 2 following lines when the meteograms actually start working.
            url_tag = epoch + '/' + meteogram_id[0] + '.png'
            image = s.get('https://images.weatherbell.net/meteogram/ecmwf-ensemble/KBLU/indiv_qpf/' + url_tag)
            #image = s.get(old_image_url)
            if image.status_code == 200:
                with open(os.path.join(img_dir, "Precip_Image", file_date + '.png'), "wb") as f:
                    f.write(image.content)
                    print("WxBell Image " + file_date + ".png Created")
            else:
                print("Wxbell Image Download Failed")
        except:
            print("Wx Bell does not have an image generated for " + datetime.now().strftime("%Y-%m-%d") + " 00Z")
            pass

        if get_GIF == True:
            try:
                nam_nest_payload = {
                    "action": "forecast",
                    "domain": "norcal",
                    "init": epoch_06Z,
                    "start": nam_ref_id[0],
                    "param": "refc",
                    "product": "nam-nest-conus",
                    "transform": "1",
                    "type": "model"
                }
                gif = s.post(req_url_gif, json=nam_nest_payload)
                if gif.status_code == 200:
                    with open(os.path.join(curDir, "Precip_Image", file_date + '.gif'), "wb") as f:
                        f.write(gif.content)
                        print("WxBell Image " + file_date + ".gif Created")

            except:
                print("Wx Bell does not have an gif generated for " + datetime.now().strftime("%Y-%m-%d") + " 06Z")
                pass



    return file_date + '.png' #change this back to '.png' when wxbell's meteograms start working again.

# def plotWheel(df):
#     # Create colors
#     blue, red, green = [plt.cm.Blues, plt.cm.Reds, plt.cm.Greens]
#     group_size = [10, 10, 10, 10]
#     explode = (0.05, 0.05, 0.05, 0.05)
#
#     #This just is a way to get the number of items in the list, and for each item make the size of the pie == 10
#     subgroup_size = list(map(lambda x: 10,df.iloc[1:8].values.flatten()))
#     explode_subgroup = list(map(lambda x: 0.05,df.iloc[1:8].values.flatten()))
#     # First Ring (outside)
#     fig, ax = plt.subplots()
#     ax.axis('equal')
#     colors = (df.iloc[1:6].values.flatten() / df.iloc[1:8].values.flatten().max())**4
#     mypie, _ = ax.pie(group_size, radius=1.3 - 0.3, labels=df.iloc[1].values, labeldistance=0.7, colors=[red(0.6), blue(0.6), green(0.6), green(0.3)], explode=explode)
#
#     # Second Ring (Inside)
#     mypie2, _ = ax.pie(subgroup_size, radius=1.3, labels=df.iloc[1:8].values.flatten(), labeldistance=0.8,
#                        colors=list(map(lambda x: red(x), colors)))
#
#     plt.setp(mypie, width=0.4, edgecolor='white')
#     plt.setp(mypie2, width=0.3, edgecolor='white')
#     plt.margins(0, 0)
#     plt.show()
#     return


def plot(df, city_code, df_historical_city, long_name):
    pd.options.mode.chained_assignment = None  # Turns off a warning that we are copying a dataframe
    # just plot the max temperatures for a given city
    keep_cols = [col for col in df.columns if city_code+'_max' in col or 'high_aw' in col or 'high_nws' in col]
    df = df[keep_cols]

    # df is a COPY of the df. Any changes we do in here MUST be returned to __main__ if we want to use it again.
    df.rename(columns={city_code+'_max_gfs': 'GFS', city_code+'_max_ecmwf': 'EURO',
                       city_code + '_max_gfs-ens-bc': 'GFS_bc',
                       city_code + '_max_ecmwf-eps': 'EURO_EPS',
                       'high_nws': 'NWS', 'high_aw': 'PCWA'}, inplace=True)

    # df is a COPY of the df. Any changes we do in here MUST be returned to __main__ if we want to use it again.
    df['PCWA'].fillna(df[['EURO_EPS', 'EURO_EPS', 'EURO_EPS', 'GFS_bc', 'GFS']].mean(axis=1), inplace=True)

    # for some reason, the NWS data was being stored as a string.
    df['NWS'] = df['NWS'].astype(float)

    for model in df.columns:
        plt.style.use('seaborn-darkgrid')
        my_dpi = 96
        plt.figure(figsize=(640 / my_dpi, 225 / my_dpi), dpi=my_dpi)
        ax = plt.gca()  # Get Current Axis.is
        COL = MplColorHelper('jet', df_historical_city["high_normal"] - 15, df_historical_city["high_normal"] + 15)

        xfmt = mdates.DateFormatter('%a\n%#m/%#d')
        line_color = {'GFS':'darkgreen', 'GFS_bc': 'limegreen', 'EURO': 'darkviolet',
                      'EURO_EPS': 'violet', 'NWS': 'firebrick', 'PCWA': 'orange'}

        line_style = {'GFS': '-', 'GFS_bc': '--', 'EURO': '-',
                      'EURO_EPS': '--', 'NWS': ':', 'PCWA': '-'}

        # multiple line plot
        column_num = 0

        for column in df.columns:
            plt.plot(df.index, df[column], marker='', color=line_color[column], linestyle = line_style[column], linewidth=1, alpha=0.7)

            column_num += 1
            if column == model:
                for x,y in zip(df.index,df[column].values):
                    x = mdates.date2num(x)
                    if not np.isnan(y):
                        ax.annotate('{}'.format(int(y)), xy=(x,y), xytext=(0,5), weight='bold',
                                    ha='center', color='red', textcoords='offset points')

        # Now re do the interesting curve, but biger with distinct color

        plt.legend(loc='upper center', ncol=6, prop={'size': 6})

        plt.plot(df.index, df[model], markerfacecolor = 'black', color=line_color[model], linestyle = line_style[model], linewidth=4, alpha=0.7, zorder=2)
        plt.scatter(df.index, df[model], color=COL.get_rgb(df[model]), edgecolors='black', zorder=3)
        x_min, x_max, y_min, y_max = plt.axis()
        plt.axis((x_min,x_max, int(math.floor(y_min / 5.0)) * 5, int(math.floor((y_max + 10) / 10.0)) * 10))
        plt.xticks(df.index, rotation=0, fontsize=8)
        plt.gcf().subplots_adjust(bottom=0.15)
        # plt.gcf().subplots_adjust(top=0)
        plt.gcf().subplots_adjust(left=0.1)
        plt.ylabel('Forecast High Temp')
        plt.title(long_name + ' Max Temp Forecast')
        ax.text(1, 1, f"Normal High: {df_historical_city.iloc[0]['high_normal']}",
                horizontalalignment='right',
                verticalalignment='bottom', weight='bold',
                transform=ax.transAxes, color='red')


        ax.xaxis.set_major_formatter(xfmt)
        # ax.plot(([df.index[1]]),[df[city_code+'_max_gfs'][1]], marker = 'o', color='black')

        # ax.annotate(str(int(df[city_code+'_max_gfs'][1])), xy=(mdates.date2num(df.index[1]),df[city_code+'_max_gfs'][1]), xytext=(mdates.date2num(df.index[1]),df['KSAC_max_gfs'][1]+3), color='r')

        # Change xlim
        # plt.xlim(0, 12)
        img_dir = os.path.join(os.path.sep, 'home', 'smotley', 'images', 'weather_email')
        if platform.system() == 'Windows':
            # If we're on windows, just save it to the current folder.
            img_dir = os.path.dirname(os.path.realpath(__file__))

        # And add a special annotation for the group we are interested in
        # plt.text(100.2, df.KSAC_max_gfs.tail(1), 'Mr Orange', horizontalalignment='left', size='small', color='orange')
        if model == 'PCWA':
            plt.savefig(os.path.join(img_dir,city_code + '_' + model +'.png'))
            #plt.show()
        plt.clf()
        plt.close()
    pd.options.mode.chained_assignment = 'warn'  # Turn warning back on
    return df


def create_merged_png():
    img_dir = os.path.join(os.path.sep, 'home', 'smotley', 'images', 'weather_email')
    if platform.system() == 'Windows':
        # If we're on windows, just save it to the current folder.
        img_dir = os.path.dirname(os.path.realpath(__file__))
    images = [Image.open(os.path.join(img_dir, x)) for x in ['KSAC_PCWA.png', 'KBUR_PCWA.png',
                                      'KPDX_PCWA.png', 'KLAS_PCWA.png',
                                      'KSEA_PCWA.png', 'KPHX_PCWA.png']]
    widths, heights = zip(*(i.size for i in images))

    max_width = max(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (max_width*2, max_height*3))  # Make a new image that's 2 col x 3 col

    x_offset = 0
    y_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, y_offset))
        x_offset += im.size[0]
        if x_offset >= im.size[0]*2:
            x_offset = 0
            y_offset += im.size[1]


    new_im.save(os.path.join(img_dir,'All_Cities.png'))
    return


if __name__ == "__main__":
    main()
