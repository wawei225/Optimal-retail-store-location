## Optimal Retail Store Location

This is a project for Coursera Data Science Certificate, supported by IBM.

This project aims at, given a set of potential locations, predicting the customer rating of a coffee shop in downtown Vancouver, Canada. 

![alt text](https://github.com/wawei225/Optimal-retail-store-location/blob/master/Capture.PNG?raw=true)

## Data sources

- **Foursquare**: venue information, customer rating, price tier, type of coffee shops, number of coffee shops in the designated area
- **Google API**: store location information, distance from Skytrain stations
- **Statistic Canada and censusmapper.ca**: population, average household income, population density of the dissemination area

## Key Approach

1. Segment the targeted area into sub-area
2. Retrieve and organize relevent information for each sub-area
3. Explore coffee shops through clustering
4. Implement supervised learning algorithms to predict customer rating
5. Perform prediction for the selected potential locations

## Things I learned from this project:

- Evaluate and plan the solution from a grand view, considering various possibilities and limitations to deliver a solution 
- Search and organize data with various different formats from online sources 
- Be familir with problem-solving pipeline

## Improvements and Limitations

- Prediction could be more informative if given revenue information for each shop
- Modelling could be improved with feature engineering and algorithms such as LTSM or LGBM
- The use of user mobility or traffic flow information could provide improve the model 

