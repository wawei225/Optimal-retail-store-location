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
Applying analysis within each cluster
"""
#===============================================================================
# Splitting training and test dataset  70% training
#===============================================================================
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

analysis_df = pd.read_csv('cluster_org_df.csv')

analysis_df2 = analysis_df.drop(columns=['neighborhood_coordination', 'venue_ID',
                          'venue_name', 'venue_address',
                          'google_address', 'postal code',
                          'category_name', 'lat', 'lng', 'DA', 'rating_class','cluster_label'])


# create training and test dataset

y = analysis_df2.venue_rating  #venue_rating is the response 
x = analysis_df2.drop('venue_rating', axis = 1)

#x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


#===============================================================================
# OLS regression with cross validation on training dataset
#===============================================================================

from sklearn.metrics import mean_squared_error
from math import sqrt

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression


# cross validation on the dataset
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for train_index, test_index in kf.split(x, y):
  # specific ".loc" syntax for working with dataframes
  x_train, x_test = x.loc[train_index], x.loc[test_index]
  y_train, y_test = y[train_index], y[test_index]

#fit linear model with cross validated train and test data
  
lm = LinearRegression()
lm_fit = lm.fit(x_train,y_train)

rmse_train_lm = sqrt(mean_squared_error(y_train, lm_fit.predict(x_train)))
rmse_test_lm = sqrt(mean_squared_error(y_test, lm_fit.predict(x_test)))

print('lm train RMSE:', rmse_train_lm)
print('lm test RMSE:', rmse_test_lm)


#===============================================================================
# There is no available package for stepwise regression; hence the variable 
# selection is run in R. The suggested variables are 
# type, venue_price_tier, dist_to_burrard, dist_to_van_city, dist_to_granville
# avg_total_household_income, and pop_density_sq_km
#
# Then a OLS regression with cross validation is run with these variables
#===============================================================================
import statsmodels.api as sm

x2 = analysis_df[['type', 'venue_price_tier', 'dist_to_burrard', 
                  'dist_to_van_city', 'shops_in_area',
                  'avg_total_household_income', 'num_dwellings','venue_rating']]

y2 = x2.venue_rating
x_x2 = x2.drop('venue_rating', axis = 1)

for train_index2, test_index2 in kf.split(x_x2, y2):
  # specific ".loc" syntax for working with dataframes
  x_train2, x_test2 = x_x2.loc[train_index2], x_x2.loc[test_index2]
  y_train2, y_test2 = y2[train_index2], y2[test_index2]

# ols model with intercept added to predictor
lm2 = sm.OLS(y_train2, x_train2)

# fitted model and summary
lm2_fit = lm2.fit()


rmse_train_lm2 = sqrt(mean_squared_error(y_train2, lm2_fit.predict(x_train2)))
rmse_test_lm2 = sqrt(mean_squared_error(y_test2, lm2_fit.predict(x_test2)))

print('lm2 train RMSE:', rmse_train_lm2)
print('lm2 test RMSE:', rmse_test_lm2)


#===============================================================================
# LASSO
#===============================================================================


from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.preprocessing import scale

grid = 10 ** np.linspace(3,-2,100)

lasso_cv = LassoCV(alphas=grid, max_iter=100000, normalize=True, cv=5)
lasso_cv.fit(x_train, y_train)
lasso_cv.alpha_

lasso = Lasso(alpha=lasso_cv.alpha_, normalize=True, max_iter=10000)
lasso_fit = lasso.fit(x_train, y_train)

rmse_train_lasso = sqrt(mean_squared_error(y_train, lasso_fit.predict(x_train)))
rmse_test_lasso = sqrt(mean_squared_error(y_test, lasso_fit.predict(x_test)))

print('lasso train RMSE:', rmse_train_lasso)
print('lasso test RMSE:', rmse_test_lasso)


#===============================================================================
# Bagging
#===============================================================================
from sklearn.ensemble import RandomForestRegressor

bagging = RandomForestRegressor(max_features=10, random_state=42)
bagging_fit = bagging.fit(x_train, y_train)


rmse_train_bagging = sqrt(mean_squared_error(y_train, bagging_fit.predict(x_train)))
rmse_test_bagging = sqrt(mean_squared_error(y_test, bagging_fit.predict(x_test)))

print('bagging train RMSE:', rmse_train_bagging)
print('bagging test RMSE:', rmse_test_bagging)


#===============================================================================
# RandomForest
#===============================================================================

rf = RandomForestRegressor(max_features=6, random_state=42, n_estimators=100)
rf_fit = rf.fit(x_train, y_train)


rmse_train_rf = sqrt(mean_squared_error(y_train, rf_fit.predict(x_train)))
rmse_test_rf = sqrt(mean_squared_error(y_test, rf_fit.predict(x_test)))

print('random forest train RMSE:', rmse_train_rf)
print('random forest test RMSE:', rmse_test_rf)

#===============================================================================
# Boosting
#===============================================================================

from sklearn.ensemble import GradientBoostingRegressor

boosting = GradientBoostingRegressor(n_estimators=100, learning_rate=0.01, max_depth=4, random_state=42)
boosting_fit = rf.fit(x_train, y_train)


rmse_train_boosting = sqrt(mean_squared_error(y_train, boosting_fit.predict(x_train)))
rmse_test_boosting = sqrt(mean_squared_error(y_test, boosting_fit.predict(x_test)))

print('boosting train RMSE:', rmse_train_boosting)
print('boosting test RMSE:', rmse_test_boosting)




"""
Using decision tree to determine main variables that distinguish the cluster
"""


cluster_df = analysis_df.drop(columns=['neighborhood_coordination', 'venue_ID',
                           'venue_name', 'venue_address',
                           'google_address', 'postal code',
                           'category_name', 'lat', 'lng', 'DA', 'venue_rating',
                           'grid_area_index'])

  
cat_0 = cluster_df[cluster_df['cluster_label']==0]
cat_1 = cluster_df[cluster_df['cluster_label']==1]
cat_2 = cluster_df[cluster_df['cluster_label']==2]
cat_3 = cluster_df[cluster_df['cluster_label']==3]


#===============================================================================
# category 0
#===============================================================================

y_cat0 = cat_0.rating_class  #venue_rating is the response 
x_cat0 = cat_0.drop(['rating_class','cluster_label'], axis = 1)

from sklearn.tree import DecisionTreeClassifier, export_graphviz

tree_cat0 = DecisionTreeClassifier(min_samples_leaf=5, max_depth=6)
tree_cat0.fit(x_cat0, y_cat0)


#===============================================================================
# category 1
#===============================================================================
                         
y_cat1 = cat_1.rating_class  #venue_rating is the response 
x_cat1 = cat_1.drop(['rating_class','cluster_label'], axis = 1)


tree_cat1 = DecisionTreeClassifier(min_samples_leaf=5, max_depth=6)
tree_cat1.fit(x_cat1, y_cat1)


#===============================================================================
# category 2
#===============================================================================                         
                                                  
y_cat2 = cat_2.rating_class  #venue_rating is the response 
x_cat2 = cat_2.drop(['rating_class','cluster_label'], axis = 1)


tree_cat2 = DecisionTreeClassifier(min_samples_leaf=5, max_depth=6)
tree_cat2.fit(x_cat2, y_cat2)
               
                         
#===============================================================================
# category 3
#===============================================================================                         
                                                  
y_cat3 = cat_3.rating_class  #venue_rating is the response 
x_cat3 = cat_3.drop(['rating_class','cluster_label'], axis = 1)


tree_cat3 = DecisionTreeClassifier(min_samples_leaf=5, max_depth=6)
tree_cat3.fit(x_cat3, y_cat3)


#===============================================================================
# Visualisation
#===============================================================================                         
       


pd.DataFrame({'Importance': tree_cat0.feature_importances_ * 100}, 
             index=x_cat0.columns).sort_values('Importance', 
                                 ascending=True, axis=0).plot(kind='barh', 
                                                       title='Feature Importance')


pd.DataFrame({'Importance': tree_cat1.feature_importances_ * 100}, 
             index=x_cat1.columns).sort_values('Importance', 
                                 ascending=True, axis=0).plot(kind='barh', 
                                                       title='Feature Importance')

pd.DataFrame({'Importance': tree_cat2.feature_importances_ * 100}, 
             index=x_cat2.columns).sort_values('Importance', 
                                 ascending=True, axis=0).plot(kind='barh', 
                                                       title='Feature Importance')


pd.DataFrame({'Importance': tree_cat3.feature_importances_ * 100}, 
             index=x_cat3.columns).sort_values('Importance', 
                                 ascending=True, axis=0).plot(kind='barh', 
                                                       title='Feature Importance')


"""
Potential location rating prediction using random forest
"""


potential_df = pd.read_csv('final_potential_df.csv')

potential_df = potential_df.drop(columns=['lat','lng','DA','postal_code'])

bagging_fit.predict(potential_df)

