# -*- coding: utf-8 -*-
"""
Created on Wed May  6 23:00:16 2020

@author: hwhua
"""

import numpy as np 
import pandas as pd
import requests # library to handle requests
from scipy import ndimage 
from scipy.spatial import distance_matrix 
from matplotlib import pyplot as plt 
from sklearn import manifold, datasets 
from sklearn.datasets.samples_generator import make_blobs 

import webbrowser

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

from pandas.io.json import json_normalize
# import k-means from clustering stage
from sklearn.cluster import KMeans

#!conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab
import folium # map rendering library



"""
Part 1

This part creates a function to get the coordinates of the address from 
Google API
"""

def get_coordinates(api_key, address, verbose=False):
    try:
        url = 'https://maps.googleapis.com/maps/api/geocode/json?key={}&address={}'.format(api_key, address)
        response = requests.get(url).json()
        if verbose:
            print('Google Maps API JSON result =>', response)
        results = response['results']
        geographical_data = results[0]['geometry']['location'] # get geographical coordinates
        lat = geographical_data['lat']
        lon = geographical_data['lng']
        return [lat, lon]
    except:
        return [None, None]

google_api_key = "ENTER YOUR KEY"

target_address = 'Waterfront Station, West Cordova Street, Vancouver'
van_waterfront = get_coordinates(google_api_key, target_address)
print('Coordinate of {}: {}'.format(target_address, van_waterfront))


"""
Part 2

To accurately calculate distances we need to create our grid of locations in Cartesian 2D coordinate system 
which allows us to calculate distances in meters (not in latitude/longitude degrees). 

Then we'll project those coordinates back to latitude/longitude degrees to be shown on Folium map. 

So let's create functions to convert between WGS84 spherical coordinate system (latitude/longitude degrees) 
and UTM Cartesian coordinate system (X/Y coordinates in meters).

"""

#conda install shapely
import shapely
#import shapely.geometry
#conda install -c conda-forge pyproj
import pyproj
#conda install math
import math

def lonlat_to_xy(lon, lat):
    proj_latlon = pyproj.Proj(proj='latlong',datum='WGS84')
    proj_xy = pyproj.Proj(proj="utm", zone=33, datum='WGS84')
    xy = pyproj.transform(proj_latlon, proj_xy, lon, lat)
    return xy[0], xy[1]

def xy_to_lonlat(x, y):
    proj_latlon = pyproj.Proj(proj='latlong',datum='WGS84')
    proj_xy = pyproj.Proj(proj="utm", zone=33, datum='WGS84')
    lonlat = pyproj.transform(proj_xy, proj_latlon, x, y)
    return lonlat[0], lonlat[1]

def calc_xy_distance(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    return math.sqrt(dx*dx + dy*dy)

print('Coordinate transformation check')
print('-------------------------------')
print('Waterfront Station longitude={}, latitude={}'.format(van_waterfront[1], van_waterfront[0]))
x, y = lonlat_to_xy(van_waterfront[1], van_waterfront[0])
print('Waterfront Station UTM X={}, Y={}'.format(x, y))
lo, la = xy_to_lonlat(x, y)
print('Waterfront Station longitude={}, latitude={}'.format(lo, la))

van_waterfront_x, van_waterfront_y = lonlat_to_xy(van_waterfront[1], van_waterfront[0])
"""
Part 5

adding distance to all the skytrain station
"""


#-------------------------------------------------------------------------------
# Extracting skytrain coordination

burrard = 'Burrard Station, Vancouver'
burrard_coord = get_coordinates(google_api_key, burrard)
burrard_coord_x, burrard_coord_y = lonlat_to_xy(burrard_coord[1], burrard_coord[0])

van_city = 'Vancouver City Centre Station, Vancouver'
van_city_coord = get_coordinates(google_api_key, van_city)
van_city_coord_x, van_city_coord_y = lonlat_to_xy(van_city_coord[1], van_city_coord[0])

granville = 'Granville Station, Vancouver'
granville_coord = get_coordinates(google_api_key, granville)
granville_coord_x, granville_coord_y = lonlat_to_xy(granville_coord[1], granville_coord[0])

yaletown = 'Yaletown-Roundhouse Station, Vancouver'
yaletown_coord = get_coordinates(google_api_key, yaletown)
yaletown_coord_x, yaletown_coord_y = lonlat_to_xy(yaletown_coord[1], yaletown_coord[0])

chinatown = 'Stadium-Chinatown Station, Vancouver'
chinatown_coord = get_coordinates(google_api_key, chinatown)
chinatown_coord_x, chinatown_coord_y = lonlat_to_xy(chinatown_coord[1], chinatown_coord[0])


# =============================================================================
#read csv and create distance
venue_df = pd.read_csv('potential_location.csv')

venue_df['dist_to_burrard'] = ""
venue_df['dist_to_van_city'] = ""
venue_df['dist_to_granville'] = ""
venue_df['dist_to_yaletown'] = ""
venue_df['dist_to_chinatown'] = ""
venue_df['dist_to_waterfront'] = ""


for i in range(len(venue_df['lat'])):
  temp_lat = venue_df['lat'][i]
  temp_lng = venue_df['lng'][i]
  temp_lat_x, temp_lng_y = lonlat_to_xy(temp_lng, temp_lat)
  
  
  temp_bur_dist = calc_xy_distance(burrard_coord_x, burrard_coord_y, temp_lat_x, temp_lng_y)
  temp_vanc_dist = calc_xy_distance(van_city_coord_x, van_city_coord_y, temp_lat_x, temp_lng_y)
  temp_gran_dist = calc_xy_distance(granville_coord_x, granville_coord_y, temp_lat_x, temp_lng_y)
  temp_yale_dist = calc_xy_distance(yaletown_coord_x, yaletown_coord_y, temp_lat_x, temp_lng_y)
  temp_china_dist = calc_xy_distance(chinatown_coord_x, chinatown_coord_y, temp_lat_x, temp_lng_y)
  temp_water_dist = calc_xy_distance(van_waterfront_x, van_waterfront_y, temp_lat_x, temp_lng_y)
   
  venue_df['dist_to_burrard'][i] = temp_bur_dist
  venue_df['dist_to_van_city'][i] = temp_vanc_dist
  venue_df['dist_to_granville'][i] = temp_gran_dist
  venue_df['dist_to_yaletown'][i] = temp_yale_dist
  venue_df['dist_to_chinatown'][i] = temp_china_dist
  venue_df['dist_to_waterfront'][i] = temp_water_dist
# =============================================================================

venue_df.to_csv (r'C:\Users\hwhua\Desktop\coursera\spyder\organised\final_potential_df.csv', index = False, header=True)  


"""
Part 6

Visualisation

Map remaining venues
"""

potential_location_df = pd.read_csv('potential_location.csv')



map_potential = folium.Map(location= van_waterfront, zoom_start=13)
folium.Marker(van_waterfront, popup='Waterfront Station').add_to(map_potential)

# add markers to map
for lat, lng in zip(potential_location_df['lat'], potential_location_df['lng']):
    folium.CircleMarker(
        [lat, lng],
        radius=3,        
        color='black',
        fill=True,
        fill_color='#000000',
        fill_opacity=1).add_to(map_potential)  
    
map_potential.save("potential_location.html")
webbrowser.open("potential_location.html")
