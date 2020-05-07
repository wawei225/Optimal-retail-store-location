# -*- coding: utf-8 -*-
"""
Created on Tue May  5 20:45:25 2020

@author: hwhua
"""

"""
Data analysis

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
Import data
"""


ori_df = pd.read_csv('final_org_df.csv')

df = ori_df.drop(columns=['neighborhood_coordination', 'venue_ID',
                          'venue_name', 'venue_address',
                          'google_address', 'postal code',
                          'category_name', 'lat', 'lng', 'DA'])
df = df.reset_index()


"""

Clustering

"""

#===============================================================================
# K-means clustering
#===============================================================================

# set number of clusters
kclusters = 4

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(df)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 

cluster_df = ori_df

cluster_df.insert(0,'cluster_label', kmeans.labels_)

#cluster_df.to_csv (r'C:\Users\hwhua\Desktop\coursera\spyder\organised\cluster_org_df.csv', index = False, header=True)  

#===============================================================================
# Visualisation
#===============================================================================

# get waterfront coordination

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



# create map
map_clusters = folium.Map(location=van_waterfront, zoom_start=13)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(cluster_df['lat'], cluster_df['lng'], cluster_df['neighborhood_coordination'], cluster_df['cluster_label']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=3,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters

map_clusters.save("map_clusters.html")
webbrowser.open("map_clusters.html")


#create map with station

station_df = pd.read_csv('station_coord.csv')

for lat, lon in zip(station_df['lat'], station_df['lng']):
    folium.CircleMarker(
        [lat, lon],
        radius=8,
        color='black',
        fill=True,
        fill_color='000000',
        fill_opacity=0.7).add_to(map_clusters)


map_clusters.save("map_clusters2.html")
webbrowser.open("map_clusters2.html")


##########################
# Add potential location #
##########################

potential_location_df = pd.read_csv('potential_location.csv')

#map_potential = folium.Map(location= van_waterfront, zoom_start=13)
folium.Marker(van_waterfront, popup='Waterfront Station').add_to(map_clusters)

# add markers to map
for lat, lng in zip(potential_location_df['lat'], potential_location_df['lng']):
    folium.CircleMarker(
        [lat, lng],
        radius=8,        
        color='#D2691E',
        fill=True,
        fill_color='#D2691E',
        fill_opacity=1).add_to(map_clusters)  
    
map_clusters.save("cluster_potential_location.html")
webbrowser.open("cluster_potential_location.html")

#===============================================================================
# DBSCAN clustering
#===============================================================================
from sklearn.cluster import DBSCAN

clustering = DBSCAN(eps=100, min_samples=2).fit(df)
clustering.labels_


ori_df = ori_df.drop(columns = ['Cluster Label'])

DBSCAN_df = ori_df

DBSCAN_df.insert(0,'Cluster Label', clustering.labels_)


DBSCAN_map = folium.Map(location=van_waterfront, zoom_start=13)


DBSCAN_clusters = 30
x = np.arange(DBSCAN_clusters)
ys = [i + x + (i*x)**2 for i in range(DBSCAN_clusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]


# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(DBSCAN_df['lat'], DBSCAN_df['lng'], DBSCAN_df['neighborhood_coordination'], DBSCAN_df['Cluster Label']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(DBSCAN_map)
       
DBSCAN_map.save("DBSCAN_map.html")
webbrowser.open("DBSCAN_map.html")




