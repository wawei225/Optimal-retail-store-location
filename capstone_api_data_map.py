# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 08:38:46 2020

@author: hwhua
"""

import requests # library to handle requests
import pandas as pd # library for data analsysis
import numpy as np # library to handle data in a vectorized manner
import random # library for random number generation
import json

#conda install -c conda-forge geopy

from geopy.geocoders import Nominatim # module to convert an address into latitude and longitude values

# libraries for displaying images
from IPython.display import Image 
from IPython.core.display import HTML 
    
# tranforming json file into a pandas dataframe library
from pandas.io.json import json_normalize

#!conda install -c conda-forge folium=0.5.0 --yes
import folium # plotting library

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

#-------------------------------------------------------------------------------


"""
Let's create a hexagonal grid of cells: we offset every other row, 
and adjust vertical row spacing so that every cell center is equally
distant from all it's neighbors.

The purpose of this part is to create neighborhoods 

Equally spaced, within 1.5 km from Waterfront Station

"""

van_waterfront_x, van_waterfront_y = lonlat_to_xy(van_waterfront[1], van_waterfront[0]) # City center in Cartesian coordinates


k = math.sqrt(3) / 2 # Vertical offset for hexagonal grid cells
x_min = van_waterfront_x - 1500
x_step = 200
y_min = van_waterfront_y -200
y_step = 200 * k 

grid_latitudes = []
grid_longitudes = []
grid_distances_from_center = []
xs = []
ys = []
for i in range(0, int(20/k)):
    y = y_min + i * y_step
    x_offset = 300 if i%2==0 else 0
    for j in range(0, 21):
        x = x_min + j * x_step + x_offset
        distance_from_center = calc_xy_distance(van_waterfront_x, van_waterfront_y, x, y)
        if (distance_from_center <= 1501):
            lon, lat = xy_to_lonlat(x, y)
            grid_latitudes.append(lat)
            grid_longitudes.append(lon)
            grid_distances_from_center.append(distance_from_center)
            xs.append(x)
            ys.append(y)
            
print(len(grid_latitudes), 'candidate neighborhood centers generated.')



#-------------------------------------------------------------------------------
# mapping the grid

map_waterfront = folium.Map(location= van_waterfront, zoom_start=13)
folium.Marker(van_waterfront, popup='Waterfront Station').add_to(map_waterfront)
for lat, lon in zip(grid_latitudes, grid_longitudes):
    #folium.CircleMarker([lat, lon], radius=2, color='blue', fill=True, fill_color='blue', fill_opacity=1).add_to(map_berlin) 
    folium.Circle([lat, lon], radius=100, color='blue', fill=False).add_to(map_waterfront)
    #folium.Marker([lat, lon]).add_to(map_berlin)

map_waterfront.save("map_waterfront_grid.html")

import webbrowser
webbrowser.open("map_waterfront_grid.html")

#-------------------------------------------------------------------------------

"""
Create a function to get address given latitude and longitude;
then, use this function to get corresponded coordinate of the grid
"""

def get_address(api_key, latitude, longitude, verbose=False):
    try:
        url = 'https://maps.googleapis.com/maps/api/geocode/json?key={}&latlng={},{}'.format(api_key, latitude, longitude)
        response = requests.get(url).json()
        if verbose:
            print('Google Maps API JSON result =>', response)
        results = response['results']
        address = results[0]['formatted_address']
        return address
    except:
        return None

addr = get_address(google_api_key, van_waterfront[0], van_waterfront[1])
print('Reverse geocoding check')
print('-----------------------')
print('Address of [{}, {}] is: {}'.format(van_waterfront[0], van_waterfront[1], addr))

print('Obtaining location addresses: ', end='')
addresses = []
for lat, lon in zip(grid_latitudes, grid_longitudes):
    address = get_address(google_api_key, lat, lon)
    if address is None:
        address = 'NO ADDRESS'
    address = address.replace(', Canada', '') # We don't need country part of address
    addresses.append(address)
    print(' .', end='')
print(' done.')


#-------------------------------------------------------------------------------
# tranform into a pandas datafram

df_locations = pd.DataFrame({'Address': addresses,
                             'Latitude': grid_latitudes,
                             'Longitude': grid_longitudes,
                             'X': xs,
                             'Y': ys,
                             'Distance from center': grid_distances_from_center})

new_df_locations = df_locations[~df_locations.Address.str.contains("Unnamed Road")]

df_locations.to_pickle('./locations.pkl')


# creating column of coordination as a string
df_locations['lat_str'] = df_locations['Latitude'].astype(str)
df_locations['lng_str'] = df_locations['Longitude'].astype(str)

df_locations['coordination_str'] = df_locations[['lat_str', 'lng_str']].apply(lambda x: ', '.join(x), axis = 1)



"""

Part 3

This part will retrive venues within the neighborhood defind in the previous part

1. a function to get the venue given lng and lat
  1.1 this function will also retrive relevant information such as rating
2. apply this function to all observations/neighborhoods from previous part
3. combined all the retrived venues to a pandas dataframe
"""

# background info for foursquare
import foursquare

CLIENT_ID = 'HCPTWG11IOQ5NKHQU3QN2UJ1Q5ZG1SBZZNA254ZHORPAWZFU' # your Foursquare ID
CLIENT_SECRET = 'FSBOYQ1TLP1Q0C54C0ZZ504RBDOPAQYZJFKU0BP5QWARIPYU' # your Foursquare Secret
VERSION = '20200331' # Foursquare API version

center_latitude = van_waterfront[0]
center_longitude = van_waterfront[1]

client = foursquare.Foursquare(client_id = CLIENT_ID, client_secret = CLIENT_SECRET)

categories = "5e18993feee47d000759b256,4bf58dd8d48988d1e0931735,5665c7b9498e7d8a4f2c0f06,5665c7b9498e7d8a4f2c0f06"



# from https://stackoverflow.com/questions/16003408/python-dict-get-with-multidimensional-dict

from functools import reduce

def chained_get(dct, *keys):
    SENTRY = object()
    def getter(level, key):
        return 'NA' if level is SENTRY else level.get(key, SENTRY)
    return reduce(getter, keys, dct)
  
 
# this function will get info of venues
    
def get_venue_info(coordination, categories, radius = 120, limit = 50):

  venues_within_range = client.venues.search(params={
                             'll': coordination,
                             'radius': radius,
                             'categoryId': categories,
                             'limit': limit})
  venues_id_list = [venues_within_range['venues'][i]['id'] for i in range(len(venues_within_range['venues']))]


  final_venue_df = pd.DataFrame(columns = ['neighborhood_coordination','venue_ID','venue_name', 'venue_address', 'category_name',
                                           'venue_rating', 'venue_price_tier',
                                           'lat', 'lng'])
  
  temp_venue_df = pd.DataFrame(columns = ['neighborhood_coordination','venue_ID','venue_name', 'venue_address', 'category_name',
                                          'venue_rating', 'venue_price_tier',
                                          'lat', 'lng'])
  
  temp_venue = {}
  
  for i in range(len(venues_id_list)):
    temp_venue[i] = client.venues(venues_id_list[i])
  
    temp_venue_df = temp_venue_df.append(
                            {
                              'neighborhood_coordination': coordination,
                              'venue_name':chained_get(temp_venue[i], 'venue', 'name'),
                              'venue_ID':chained_get(temp_venue[i], 'venue', 'id'),
                              'venue_address':chained_get(temp_venue[i], 'venue', 'location','address'),
                              'venue_rating':chained_get(temp_venue[i], 'venue', 'rating'),
                              'venue_price_tier':chained_get(temp_venue[i], 'venue', 'price', 'tier'),
                              'category_name':chained_get(temp_venue[i], 'venue', 'categories')[0]['name'],
                              'lat':chained_get(temp_venue[i], 'venue', 'location','lat'),
                              'lng':chained_get(temp_venue[i], 'venue', 'location','lng')},
                              ignore_index=True)
    
  final_venue_df = pd.concat([final_venue_df, temp_venue_df])
    
  return(final_venue_df)


#-------------------------------------------------------------------------------
# looping over all the coordination of the grid

all_venue_list = pd.DataFrame()  

for i in range(len(df_locations['coordination_str'])):
  temp_coordination = get_venue_info(df_locations['coordination_str'][i],categories,radius=125,limit=10)
  all_venue_list = pd.concat([all_venue_list,temp_coordination]).drop_duplicates().reset_index(drop=True)
  

#-------------------------------------------------------------------------------
# drop duplicate venue ID
org_venue_list = all_venue_list.drop_duplicates(subset=['venue_ID'], keep='first')

# write out .csv  
org_venue_list.to_csv (r'C:\Users\hwhua\Desktop\coursera\spyder\org_venue_list2.csv', index = False, header=True)  



"""
Part 4

Visualisation

Map of all the venues
"""

# add markers to map
for lat, lng in zip(org_venue_list['lat'], org_venue_list['lng']):
    folium.CircleMarker(
        [lat, lng],
        radius=3,        
        color='red',
        fill=True,
        fill_color='#FF0000',
        fill_opacity=0.7).add_to(map_waterfront)  
    
map_waterfront.save("map_vancouver_with_venue.html")
webbrowser.open("map_vancouver_with_venue.html")




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
venue_df = pd.read_csv('org_venue_list2.csv')

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

venue_df.to_csv (r'C:\Users\hwhua\Desktop\coursera\spyder\final_df.csv', index = False, header=True)  


"""
Part 6

Visualisation

Map remaining venues
"""

final_venue_df = pd.read_csv('final_org_df.csv')
# add markers to map
for lat, lng in zip(final_venue_df['lat'], final_venue_df['lng']):
    folium.CircleMarker(
        [lat, lng],
        radius=3,        
        color='black',
        fill=True,
        fill_color='#000000',
        fill_opacity=1).add_to(map_waterfront)  
    
map_waterfront.save("map_vancouver_final_venue.html")
webbrowser.open("map_vancouver_final_venue.html")
