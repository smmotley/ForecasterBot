# Author: Shane Motley
# Modified: 8/18/18
# Added to version control
from noaa_sdk import noaa
import pandas as pd
from datetime import datetime
from datetime import timedelta
from dateutil import parser
import pytz
from pandas.io.json import json_normalize
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
from matplotlib import cm
import time
import numpy as np
import math
import os
import sqlite3
from pathlib import Path
from sqlalchemy import create_engine

plt.interactive(False)
KEYS = pd.read_csv(os.path.join(Path(__file__).parents[2], 'userkeys.config'))
n = noaa.NOAA()


class FORECASTDAY(object):
    def __init__(self, outlet):
        self.day0 = {}
        self.day1 = {}
        self.day2 = {}
        self.day3 = {}
        self.day4 = {}
        self.day5 = {}
        self.day6 = {}
        self.day7 = {}
        self.day8 = {}
        self.day9 = {}
        self.day10 = {}


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
                     {'KSAC': {'latlng': (38.61, -121.41)},
                      'KBUR': {'latlng': (34.23, -118.48)},
                      'KPDX': {'latlng': (45.53, -122.64)},
                      'KSEA': {'latlng': (47.65, -122.30)},
                      'KLAS': {'latlng': (36.17, -115.18)},
                      'KPHX': {'latlng': (33.45, -112.07)}
                      }}

    # yesterday = historical('KSAC')
    # df_hourly = hourly(n, convert_to_daily=True)
    df_models = stormVista(list(locations['city_code']))

    for location in locations['city_code']:
        df_daily = daily(n, locations['city_code'][location]['latlng'])
        df_final = pd.merge(df_models.filter(like=location + '_max', axis='columns'), df_daily,
                            left_on=[df_models.index.month, df_models.index.day],
                            right_on=[df_daily.index.month, df_daily.index.day], how='left').set_index(df_models.index)

        # Must return a dataframe since we're manipulating df_final in this def
        df_maxt = plot(df_final, location)
        sql_inject(df_maxt, location)
    # df_final.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),'output.csv'), index=True,
    # columns=['KSAC_max_gfs', 'KSAC_max_gfs-ens-bc','KSAC_max_ecmwf','KSAC_max_ecmwf-eps','high.nws','high.wu'])


def sql_inject(df, location):
    db_path = os.path.curdir + '/database.sqlite3'
    # db_path = os.path.join(os.path.dirname(__file__), 'database.sqlite3')
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    df.index = df.index.tz_convert(None)
    print(location)
    # df.columns = [str(col) + '_Tmax' for col in df.columns]
    df['city_code'] = location
    df['date_created'] = df.index[0]
    df['date_valid'] = df.index
    df.to_sql('d2', conn, if_exists='append', index=False)
    return


def hourly(n, latlng, convert_to_daily):
    # 1) Grab the data from the interwebs
    res_nws = n.points_forecast(latlng[0], latlng[1], hourly=True)     # NWS POINT FORECAST
    res_wu = n.wu_forecast(latlng[0], latlng[1], hourly=True)          # Wunderground POINT FORECAST

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

    plt.show()

    # 7) Resample hourly data into daily data based off of local timezone information
    df_fcst_daily = df_fcst_hourly.resample('D')['wu_temp','nws_temp'].agg({'min','max'})

    # If you don't joint the columns names, the names will be a tuple which is difficult to access
    df_fcst_daily.columns = ['_'.join(col).strip() for col in df_fcst_daily.columns.values]
    if convert_to_daily:
        return df_fcst_daily
    return df_fcst_hourly


def daily(n, latlng):
    # 1) Get json data from NWS and WU response. We are getting daily data here, so set hourly flag to FALSE
    res_nws = n.points_forecast(latlng[0], latlng[1], hourly=False)  # NWS POINT FORECAST
    res_wu = n.wu_forecast(latlng[0], latlng[1], hourly=False)  # Wunderground POINT FORECAST

    # 2) Make dataframe object from the json response
    # Make a dataframe object of the WU forecast by flattening out the json file.
    df_wu = pd.concat([pd.DataFrame(json_normalize(x)) for x in res_wu['forecast']['simpleforecast']['forecastday']], ignore_index=True)
    df_nws = pd.DataFrame(res_nws['properties']['periods'])

    # 3) Take the string object and convert to a datetime object.
    #   Set the column 'day' to the 'epoch' object in the json file.
    #   Since the NWS must match these dates, we just use WU dates.
    df_wu['day'] = [datetime.utcfromtimestamp(float(t['date']['epoch'])) for t in res_wu['forecast']['simpleforecast']['forecastday']]

    # 4) Get time zone data and convert any datetime columns to datetime objects that Pandas can read.
    # Get the timezone location for our point from Wunderground's json file.
    wu_timezone = res_wu['forecast']['simpleforecast']['forecastday'][0]['date']['tz_long']
    nws_timezone = parser.parse(res_nws['properties']['periods'][0]['startTime'])

    # Provide timezone information to our dataframe by first setting the time information to UTC,
    # then converting to the actual timezone
    df_wu['day']=df_wu['day'].dt.tz_localize(pytz.utc)
    df_wu['day'] = df_wu['day'].dt.tz_convert(wu_timezone)
    df_wu.rename(columns={'high.fahrenheit': 'high_wu', 'low.fahrenheit': 'low_wu'}, inplace=True)
    df_wu[['high_wu', 'low_wu']]=df_wu[['high_wu','low_wu']].apply(pd.to_numeric)

    # Make sure the time columns are a datetime object for Pandas to read
    df_nws['startTime'] = pd.to_datetime(df_nws['startTime'], utc=True)
    df_nws['startTime'] = df_nws['startTime'].dt.tz_convert(nws_timezone.tzinfo)
    df_nws['endTime'] = pd.to_datetime(df_nws['endTime'], utc=True)
    df_nws['endTime'] = df_nws['endTime'].dt.tz_convert(nws_timezone.tzinfo)

    # 5) NWS Only: There is no "high" or "low" column in the data, so we have to create one:
    # Instead of combining the data by a single day, the NWS provides a start and end
    # time for each period with an "isDaytime" flag. We will use the isDaytime flag to get
    # the daytime and nighttime temperaures.
    df_nws['high_nws']= df_nws[df_nws.isDaytime.isin([True])]['temperature']
    df_nws['low_nws'] = df_nws[df_nws.isDaytime.isin([False])]['temperature']

    # We want to merge the dataframes off of a unique date, but the NWS data has the same date
    # muliple times. Therefore, we will just make a "day and night" dataframe that has only one day
    # per entry, which will allow us to merge that data with the wUnderground data.
    df_nws_day = df_nws[pd.notnull(df_nws['high_nws'])]
    df_nws_night = df_nws[pd.notnull(df_nws['low_nws'])]

    # 6) Merge the two dataframes twice: Once to get the NWS high temperatures into the df and once to get the low
    # temperatures. After this is done, we have one dataframe that we can return.
    df_wu = pd.merge(df_wu, df_nws_day[['high_nws']], left_on=[df_wu.day.dt.month,df_wu.day.dt.day],
                      right_on=[df_nws_day['endTime'].dt.month, df_nws_day['endTime'].dt.day], how='left')

    # for some reason, the merge puts in a key_0, key_1 col, which needs to be deleted before we do the next merge.
    df_wu.drop(['key_0','key_1'], axis = 1 ,inplace = True)

    df_wu = pd.merge(df_wu, df_nws_night[['low_nws']], left_on=[df_wu.day.dt.month, df_wu.day.dt.day],
                        right_on=[df_nws_night['endTime'].dt.month, df_nws_night['endTime'].dt.day], how='left')

    # for some reason, the merge puts in a key_0, key_1 col, which needs to be deleted before we do the next merge.
    df_wu.drop(['key_0', 'key_1'], axis=1, inplace=True)

    df_wu.set_index('day', inplace=True)
    # Note: even though the df_wu will display in UTC if you view the df, it is still timezone aware and
    #       will contain info for the correct timezone (prove it by uncommenting the print statement below).
    # print(df_wu.index[0].day)
    return df_wu


def historical(cities):
    hist = n.historical_data((datetime.now(pytz.timezone('US/Pacific'))) + timedelta(days=-1))
    station = list(filter(lambda s: s['properties']['station'] == cities, hist['features']))[0]
    return station


def stormVista(cities):

    base_url = "https://www.stormvistawxmodels.com/"
    clientKey =  KEYS.iloc[0]['key']
    models = ["gfs", "ecmwf", "gfs-ens-bc", "ecmwf-eps"]
    # hours = np.arange(0,360,6)
    today = datetime.utcnow()
    tomorrow = (datetime.utcnow() + timedelta(days=1)).strftime("%Y%m%d")

    modelCycle = '12z'
    # General rule that the 12Z model is not out until roughly 1:00 pm PDT (20z) and the 00Z isn't avail until 1:00 am (0800Z)
    if today.hour >= 8 and today.hour <= 20:
        modelCycle = '00z'

    # Create and empty dataframe that will hold dates in the ['date'] column for 16 days.
    df_models = pd.DataFrame(data=pd.date_range(start=today.strftime("%Y%m%d"), periods=16, freq='D'), columns=['date'])

    for model in models:
        # raw = "client-files/" + clientKey + "/model-data/" + model + "/" + today + "/" + modelCycle + "/city-extraction/individual/" + regions + "_raw.csv"
        sv_min_max_all = "client-files/" + clientKey + "/model-data/" + model + "/" + today.strftime("%Y%m%d") + "/" + modelCycle + "/city-extraction/corrected-max-min_northamerica.csv"
        fileName = today.strftime("%Y%m%d") + "_" + modelCycle + "_" + model + ".csv"
        curDir = os.path.curdir
        print(curDir)
        # curDir = os.path.dirname(os.path.abspath(__file__))
        # Download file if it hasn't been downloaded yet.
        if not os.path.isfile(os.path.join(curDir,fileName)):
            try:
                df_corrected = pd.read_csv(base_url + sv_min_max_all, header=[0, 1])
                time.sleep(5)  # Wait 5 seconds before continuing (per stormvista api requirement)
                df_corrected.to_csv(os.path.join(curDir,fileName), index=False)
            except:
                return pd.DataFrame(data=pd.date_range(start=today,
                                             end=(datetime.utcnow() + timedelta(days=15)).strftime("%Y%m%d"),
                                                       freq='D'), columns=['date'])
        else:
            df_corrected = pd.read_csv(os.path.join(curDir,fileName), header=[0, 1])

        df_corrected.columns = df_corrected.columns.map('/'.join)
        df_corrected.set_index(['station/station'], inplace=True)
        dates = [c[:-4] for c in df_corrected.columns]
        df = pd.DataFrame(data=pd.date_range(start=dates[0],
                                             end=dates[-1], freq='D'), columns=['date'])
        df_mins = df_corrected[[col for col in df_corrected if 'min' in col]]
        df_maxs = df_corrected[[col for col in df_corrected if 'max' in col]]

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

        # dates = [c for c in df_corrected.columns if c[-2:] != '.1' and c != 'station']
        # df_corrected.set_index(['station_station'], inplace=True)

        # df_corrected.rename(columns = {'max' : model + '_max', 'min' : model + '_min'}, inplace=True)
        # time.sleep(5) #Wait 5 seconds before continuing (per stormvista api requirement)
        # df_raw = pd.read_csv(base_url + raw)
        # time.sleep(5)  # Wait 5 seconds before continuing (per stormvista api requirement)
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


def plot(df, city_code):
    pd.options.mode.chained_assignment = None # Turns off a warning that we are copying a dataframe
    # just plot the max temperatures for a given city
    keep_cols = [ col for col in df.columns if city_code+'_max' in col or 'high_wu' in col or 'high_nws' in col]
    df = df[keep_cols]

    # df is a COPY of the df. Any changes we do in here MUST be returned to __main__ if we want to use it again.
    df.rename(columns={city_code+'_max_gfs': 'GFS', city_code+'_max_ecmwf': 'EURO',
                       city_code + '_max_gfs-ens-bc': 'GFS-bc',
                       city_code + '_max_ecmwf-eps': 'EPS',
                       'high_nws': 'NWS', 'high_wu': 'PCWA'}, inplace=True)

    # df is a COPY of the df. Any changes we do in here MUST be returned to __main__ if we want to use it again.
    df['PCWA'].fillna(df[['EPS', 'EPS', 'EPS', 'GFS-bc', 'GFS']].mean(axis=1), inplace = True)

    for model in df.columns:
        plt.style.use('seaborn-darkgrid')
        my_dpi = 96
        plt.figure(figsize=(640 / my_dpi, 225 / my_dpi), dpi=my_dpi)
        ax = plt.gca()  # Get Current Axis
        COL = MplColorHelper('jet', 60, 110)

        xfmt = mdates.DateFormatter('%a\n%#m/%#d')
        line_color = {'GFS':'darkgreen', 'GFS-bc':'limegreen','EURO':'darkviolet',
                      'EPS':'violet','NWS':'firebrick', 'PCWA' : 'orange'}

        line_style = {'GFS': '-', 'GFS-bc': '--', 'EURO': '-',
                      'EPS': '--', 'NWS': ':', 'PCWA': '-'}

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
        # points = np.array([mdates.date2num(df.index.to_pydatetime()), df['gfs']]).T.reshape(-1, 1, 2)
        # norm = plt.Normalize(60, 110)
        # lc = LineCollection(np.concatenate([points[:-1], points[1:]], axis=1), cmap=('jet'), norm=norm)
        # lc.set_array(df['gfs'])
        # lc.set_linewidth(2)
        # plt.gca().add_collection(lc)
        plt.legend(loc='upper center', ncol=6, prop={'size': 6})

        plt.plot(df.index, df[model], markerfacecolor = 'black', color=line_color[model], linestyle = line_style[model], linewidth=4, alpha=0.7, zorder=2)
        plt.scatter(df.index, df[model], color=COL.get_rgb(df[model]), edgecolors='black', zorder=3)
        x_min, x_max, y_min, y_max = plt.axis()
        plt.axis((x_min,x_max, int(math.floor(y_min / 5.0)) * 5, int(math.floor((y_max + 10)/ 10.0)) * 10))
        plt.xticks(df.index, rotation=0, fontsize = 8)
        plt.gcf().subplots_adjust(bottom=0.15)
        # plt.gcf().subplots_adjust(top=0)
        plt.gcf().subplots_adjust(left=0.1)
        plt.ylabel('Forecast High Temp')
        plt.title(city_code + ' Max Temp Forecast')


        ax.xaxis.set_major_formatter(xfmt)
        # ax.plot(([df.index[1]]),[df[city_code+'_max_gfs'][1]], marker = 'o', color='black')

        # ax.annotate(str(int(df[city_code+'_max_gfs'][1])), xy=(mdates.date2num(df.index[1]),df[city_code+'_max_gfs'][1]), xytext=(mdates.date2num(df.index[1]),df['KSAC_max_gfs'][1]+3), color='r')

        # Change xlim
        # plt.xlim(0, 12)

        # And add a special annotation for the group we are interested in
        # plt.text(100.2, df.KSAC_max_gfs.tail(1), 'Mr Orange', horizontalalignment='left', size='small', color='orange')
        if model == 'PCWA':
            plt.savefig(city_code + '_'+ model +'.png')
            plt.show()
        plt.clf()
        plt.close()
    pd.options.mode.chained_assignment = 'warn' # Turn warning back on
    return df


if __name__ == "__main__":
    main()
