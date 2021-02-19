"""
API Wrapper for NOAA API V3
===========================
For more detailed information about NOAA API,
visit: https://forecast-v3.weather.gov/documentation

Geoencoding is made possible by Open Street Map (© OpenStreetMap contributors)
For copyright information, visit: https://www.openstreetmap.org/copyright


"""

import json
from urllib.parse import urlencode

from util import UTIL

class ACCEPT(object):
    """Encapsulate all accept strings."""

    GEOJSON = 'application/geo+json'
    JSONLD = 'application/ld+json'
    DWML = 'application/vnd.noaa.dwml+xml'
    OXML = 'application/vnd.noaa.obs+xml'
    CAP = 'application/cap+xml'
    ATOM = 'application/atom+xml'
    JSON = 'application/json'

class NOAA(UTIL):
    """Main class for getting data from NOAA."""

    DEFAULT_END_POINT = 'api.weather.gov'
    DEFAULT_END_POINT_WU = 'api.wunderground.com'
    DEFAULT_END_POINT_IA = 'mesonet.agron.iastate.edu'
    DEFAULT_USER_AGENT = 'Test (your@email.com)'
    DEFAULT_USER_KEY = '71a80ad0edde3930'

    def __init__(self, user_agent=None, accept=None, show_uri=False):
        """Constructor.

        Args:
            user_agent (str[optional]): user agent specified in the header.
            accept (str[optional]): accept string specified in the header.
            show_uri (boolean[optional]): True for showing the
                actual url with query string being sent for requesting data.
        """
        if not user_agent:
            user_agent = self.DEFAULT_USER_AGENT
        if not accept:
            accept = ACCEPT.GEOJSON


        super().__init__(
            user_agent=user_agent, accept=accept,
            show_uri=show_uri)
        self._osm = OSM()

    def get_forecasts(self, postal_code, country, hourly=True):
        """Get forecasts by postal code and country code.

        Args:
            postalcode (str): postal code.
            country (str): 2 letter country code.
            hourly (boolean[optional]): True for getting hourly forecast.
        Returns:
            list: list of weather forecasts.
        """

        lat, lon = self._osm.get_lat_lon_by_postalcode_country(postal_code, country)
        res = self.points_forecast(lat, lon, hourly)
        if 'properties' in res and 'periods' in res['properties']:
            return res['properties']['periods']
        elif 'status' in res and res['status'] == 503 and 'detail' in res:
            raise Exception('Status: {}, NOAA API Error Response: {}'.format(
                res['status'], res['detail']))
        return []

    def points(self, point, stations=False):
        """Metadata about a point.
        This is the primary endpoint for forecast information for a location.
        It contains linked data for the forecast, the hourly forecast,
        observation and other information. It also shows stations nearest to a point
        in order of distance.

        Response in this method should not be modified.
        In this way, we can keep track of changes made by NOAA through
        functional tests @todo(paulokuong) later on.

        Args:
            point (str): lat,long.
            stations (boolean): True for finding stations.
        Returns:
            json: json response from api.
        """

        if stations:
            return self.make_get_request(
                "/points/{point}/stations".format(point=point),
                end_point=self.DEFAULT_END_POINT)
        return self.make_get_request(
            "/points/{point}".format(point=point),
            end_point=self.DEFAULT_END_POINT)

    def points_forecast(self, lat, long, hourly=False):
        """Get observation data from a weather station.

        Response in this method should not be modified.
        In this way, we can keep track of changes made by NOAA through
        functional tests @todo(paulokuong) later on.

        Args:
            lat (float): latitude of the weather station coordinate.
            long (float): longitude of the weather station coordinate.
            hourly (boolean[optional]): True for getting hourly forecast.
        Returns:
            json: json response from api.
        """

        points = self.make_get_request(
            "/points/{lat},{long}".format(
                lat=lat, long=long), end_point=self.DEFAULT_END_POINT)
        uri = points['properties']['forecast']
        if hourly:
            uri = points['properties']['forecastHourly']

        return self.make_get_request(
            uri=uri, end_point=self.DEFAULT_END_POINT)

    def wu_forecast(self, lat, long, hourly=True):
        """Get observation data from a weather station.

        Response in this method should not be modified.
        In this way, we can keep track of changes made by NOAA through
        functional tests @todo(paulokuong) later on.

        Args:
            lat (float): latitude of the weather station coordinate.
            long (float): longitude of the weather station coordinate.
            hourly (boolean[optional]): True for getting hourly forecast.
        Returns:
            json: json response from api.
        """
        #day10URL = ("http://api.wunderground.com/api/71a80ad0edde3930/forecast10day/q/%s,%s.json") % (lat, lon)

        uri =  "https://api.wunderground.com/api/{token}/forecast10day/q/{lat},{long}.json".format(
                token=self.DEFAULT_USER_KEY, lat=lat, long=long)

        if hourly:
            uri = "https://api.wunderground.com/api/{token}/hourly10day/q/{lat},{long}.json".format(
                token=self.DEFAULT_USER_KEY, lat=lat, long=long)

        return self.make_get_request(
            uri=uri, end_point=self.DEFAULT_END_POINT_WU)

    def historical_data(self, date):
        uri = "http://mesonet.agron.iastate.edu/geojson/cli.py?dt={date}&fmt=geojson".format(
            date=date.strftime('%Y-%m-%d'))
        return self.make_get_request(
            uri=uri, end_point=self.DEFAULT_END_POINT_IA)




