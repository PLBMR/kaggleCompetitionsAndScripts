#process.py
#main script for processing my dataset into its canonical form.

#imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import seaborn as sns
import pickle as pkl
from sklearn import preprocessing as pp
from haversine import haversine
from geopy.distance import vincenty, great_circle

#helpers

def oneHotEncode(givenFrame,varName):
    #helper that gets us the one-hot encoding for a particular variable
    givenEncoder = pp.OneHotEncoder(sparse = True) #gives us sparse matrix
    varMat = np.array(givenFrame[varName]).reshape(-1,1) #column format
    oneHotEncodeMat = givenEncoder.fit_transform(varMat)
    return oneHotEncodeMat

def generateInteractionMat(matOne,matTwo):
    #helper script for generating the interaction matrix between two sparse
    #matrices
    interactionMat = None
    for i in xrange(matOne.shape[1]):
        matOneVec = np.squeeze(matOne[:,i].toarray())
        matOneDiagMat = sp.sparse.diags(matOneVec)
        givenIntMat = matOneDiagMat * matTwo
        if (type(interactionMat) == type(None)):
            interactionMat = givenIntMat
        else:
            interactionMat = sp.sparse.hstack((interactionMat,givenIntMat))
    return interactionMat

def generateTileEncoding(featureFrame,locType,minLocLat,
                         maxLocLat,minLocLong,maxLocLong,step):
    #helper for generating our tile encoding for a given location type
    locLat = locType + "_latitude"
    locLong = locType + "_longitude"
    #get min and max for both
    #need trainframe so as to standardize the search on both parameter sets
    #then form latitude and longitude ranges
    latRange = np.arange(minLocLat,maxLocLat,step)
    longRange = np.arange(minLocLong,maxLocLong,step)
    #then get our matrix
    tileEncodingMat =  np.zeros((featureFrame.shape[0],len(latRange)*
                                                      len(longRange)))
    #then step through our ranges
    for i in xrange(len(latRange)):
        for j in xrange(len(longRange)):
            #get our box
            lat, lon = latRange[i], longRange[j]
            x0, y0, x1, y1 = lon, lat, lon + step, lat + step
            #form our tile encoding
            condition = ((x0 <= featureFrame[locLong]) &
                         (featureFrame[locLong] <= x1) &
                         (y0 <= featureFrame[locLat]) &
                         (featureFrame[locLat] <= y1))
            tileEncoding = list(condition.astype("int"))
            tileEncodingMat[:,(i * len(longRange) + j)] = tileEncoding
    #then filter out 0 variance observationa
    #tileEncodingMat = tileEncodingMat[:,(tileEncodingMat.sum(axis = 0) > 0)]
    return sp.sparse.csr_matrix(tileEncodingMat)

def getDistanceMetricMat(givenFrame,distanceMetric,distanceMetricName):
    #helper for calculating our distance metric across our givenFrame and then
    #storing that into a csc_matrix for appending to our feature matrix
    #get distance metric calculated
    if (distanceMetric == haversine):
        givenFrame[distanceMetricName] = [
                                        distanceMetric((x[1]["pickup_latitude"],
                                                      x[1]["pickup_longitude"]),
                                                     (x[1]["dropoff_latitude"],
                                                     x[1]["dropoff_longitude"]),
                                                     miles = True) 
                                                for x in givenFrame.iterrows()]
    else: #geopy version
        givenFrame[distanceMetricName] = [
                                        distanceMetric((x[1]["pickup_latitude"],
                                                      x[1]["pickup_longitude"]),
                                                     (x[1]["dropoff_latitude"],
                                                     x[1]["dropoff_longitude"])
                                                     ).miles 
                                                for x in givenFrame.iterrows()]
    #now that we have that, reformat it into csc matrix
    distanceMat = sp.sparse.csc_matrix(np.array(
                                        givenFrame[distanceMetricName])).T
    return distanceMat

def processHolidays(holidayFrame):
    #helper that processes the data in our holiday frame
    def getMonth(row):
        #helper for getting month in a row of observations for the holiday frame
        monthVec = ["January","February","March","April","May","June","July",
                "August","September","October","November","December"]
        givenMonth = row["Date"].split(" ")[0]
        return(monthVec.index(givenMonth) + 1) 
    def getDay(row):
        givenDay = row["Date"].split(" ")[1]
        return(int(givenDay))
    holidayFrame["month"] = [getMonth(row[1]) 
                                for row in holidayFrame.iterrows()]
    holidayFrame["day"] = [getDay(row[1]) for row in holidayFrame.iterrows()]
    #drop some information from holiday frame
    holidayFrame = holidayFrame.drop(["Day"],axis = 1)
    return holidayFrame

def processWeatherFrame(weatherFrame):
    #helper for processing our weather frame
    #get weather date
    weatherFrame["datetime"] = pd.to_datetime(weatherFrame["date"],
                                                      format = "%d-%m-%Y")
    weatherFrame["day"] = weatherFrame["datetime"].dt.day
    weatherFrame["month"] = weatherFrame["datetime"].dt.month
    #get rid of some variables
    weatherFrame = weatherFrame.drop(["date","datetime"],axis = 1)
    #alter trace to 0
    weatherFrame["precipitation_t0"] = weatherFrame["precipitation"]
    weatherFrame.loc[weatherFrame.precipitation == "T","precipitation_t0"] = 0
    weatherFrame["precipitation_t0"] = weatherFrame[
                                        "precipitation_t0"].astype("float")
    weatherFrame["snow fall_t0"] = weatherFrame["snow fall"]
    weatherFrame.loc[weatherFrame["snow fall"] == "T","snow fall_t0"] = 0
    weatherFrame["snow fall_t0"] = weatherFrame["snow fall_t0"].astype("float")
    weatherFrame["snow depth_t0"] = weatherFrame["snow depth"]
    weatherFrame.loc[weatherFrame["snow depth"] == "T","snow depth_t0"] = 0
    weatherFrame["snow depth_t0"] = weatherFrame[
                                        "snow depth_t0"].astype("float")
    #then log
    weatherFrame["logPrecipitation"] = np.log(
                                        weatherFrame["precipitation_t0"] + 1)
    weatherFrame["logSnowFall"] = np.log(weatherFrame["snow fall_t0"] + 1)
    weatherFrame["logSnowDepth"] = np.log(weatherFrame["snow depth_t0"] + 1)
    return weatherFrame

def processFeatures(givenFrame,holidayFrame,noiseComplaintFrame,weatherFrame,
                    osrmFrame):
    #helper for processing our features for preparation
    #first get dow and hour into our matrix
    #create datetime objects
    givenFeatureMat = None #will alter this
    givenFrame["pickup_datetime"] = pd.to_datetime(
                                            givenFrame["pickup_datetime"])
    givenFrame["pickup_dow"] = givenFrame["pickup_datetime"].dt.dayofweek
    givenFrame["pickup_hour"] = givenFrame["pickup_datetime"].dt.hour
    #get matrices
    dowMat = oneHotEncode(givenFrame,"pickup_dow")
    hourMat = oneHotEncode(givenFrame,"pickup_hour")
    givenFeatureMat = sp.sparse.hstack((dowMat,hourMat))
    #get their interaction
    dowHourMat = generateInteractionMat(dowMat,hourMat)
    givenFeatureMat = sp.sparse.hstack((givenFeatureMat,dowHourMat))
    #get more than one passengers encoding
    givenFrame["moreThan0Passengers"] = 0
    givenFrame.loc[givenFrame["passenger_count"] > 0,"moreThan0Passengers"] = 1
    givenPassengerFeatureMat = sp.sparse.csc_matrix(
                                            givenFrame["moreThan0Passengers"]).T
    givenFeatureMat = sp.sparse.hstack((givenFeatureMat,
                                        givenPassengerFeatureMat))
    #add seasonality
    givenFrame["pickup_month"] = givenFrame["pickup_datetime"].dt.month
    monthMat = oneHotEncode(givenFrame,"pickup_month")
    givenFeatureMat = sp.sparse.hstack((givenFeatureMat,monthMat))
    #then get tile encoding of pickup and dropoff
    #currently hardcoded numbers, will update eventually with more meaningful
    #variable names
    pickupEncodingMat = generateTileEncoding(givenFrame,"pickup",40.2,41.8,
                                             -75,-72.5,.1)
    dropoffEncodingMat = generateTileEncoding(givenFrame,"dropoff",40.2,41.8,
                                             -75,-72.5,.1)
    givenFeatureMat = sp.sparse.hstack((givenFeatureMat,pickupEncodingMat))
    givenFeatureMat = sp.sparse.hstack((givenFeatureMat,dropoffEncodingMat))
    #get distance metrics into our model
    haversineMat = getDistanceMetricMat(givenFrame,haversine,"haversine")
    vincentyMat = getDistanceMetricMat(givenFrame,vincenty,"vincenty")
    gcMat = getDistanceMetricMat(givenFrame,great_circle,"gc")
    givenFeatureMat = sp.sparse.hstack((givenFeatureMat,haversineMat))
    givenFeatureMat = sp.sparse.hstack((givenFeatureMat,vincentyMat))
    givenFeatureMat = sp.sparse.hstack((givenFeatureMat,gcMat))
    #then add interactions for each of these features and time of day
    haversineHourMat = generateInteractionMat(haversineMat,hourMat)
    vincentyHourMat = generateInteractionMat(vincentyMat,hourMat)
    gcHourMat = generateInteractionMat(gcMat,hourMat)
    givenFeatureMat = sp.sparse.hstack((givenFeatureMat,haversineHourMat))
    givenFeatureMat = sp.sparse.hstack((givenFeatureMat,vincentyHourMat))
    givenFeatureMat = sp.sparse.hstack((givenFeatureMat,gcHourMat))
    #then add additional datasets
    #NYC holidays
    holidayFrame = processHolidays(holidayFrame)
    givenFrame["day"] = givenFrame["pickup_datetime"].dt.day
    givenFrame["month"] = givenFrame["pickup_datetime"].dt.month
    #merge
    givenFrame = givenFrame.merge(holidayFrame,on = ["day","month"],
                                  how = "left")
    givenFrame.loc[givenFrame.Holiday.isnull(),"Holiday"] = "No Holiday"
    #get is holiday
    givenFrame["isHoliday"] = 1
    givenFrame.loc[givenFrame.Holiday == "No Holiday","isHoliday"] = 0
    givenIsHolidayFeatureMat = sp.sparse.csc_matrix(
                                            givenFrame["isHoliday"]).T
    givenFeatureMat = sp.sparse.hstack((givenFeatureMat,
                                        givenIsHolidayFeatureMat))
    #get Noise complaints
    givenFrame = givenFrame.merge(noiseComplaintFrame,on = "id",how = "left")
    givenFrame["logNumComplaints"] = np.log(givenFrame.num_complaints + 1)
    logNumComplaintsMat = sp.sparse.csc_matrix(givenFrame["logNumComplaints"]).T
    givenFeatureMat = sp.sparse.hstack((givenFeatureMat,logNumComplaintsMat))
    #get weather
    weatherFrame = processWeatherFrame(weatherFrame)
    givenFrame = givenFrame.merge(weatherFrame,on = ["day","month"],
                                  how = "left")
    #temperature
    tempMat = sp.sparse.csc_matrix(
                givenFrame[["maximum temperature","minimum temperature",
                            "average temperature"]])
    #accumulation
    accMat  = sp.sparse.csc_matrix(
                givenFrame[["logPrecipitation","logSnowFall","logSnowDepth"]])
    givenFeatureMat = sp.sparse.hstack((givenFeatureMat,tempMat))
    givenFeatureMat = sp.sparse.hstack((givenFeatureMat,accMat))
    #then get their interaction
    tempAccMat = generateInteractionMat(tempMat,accMat)
    givenFeatureMat = sp.sparse.hstack((givenFeatureMat,tempAccMat))
    #then get osrm data
    givenFrame = givenFrame.merge(osrmFrame,on = "id",how = "left")
    #filter out data without osrm information
    osrmCondition = ((givenFrame["total_distance"].notnull()) &
                  (givenFrame["total_travel_time"].notnull()) &
                  (givenFrame["number_of_steps"].notnull()))
    givenFrame = givenFrame[osrmCondition]
    remainderIndices = list(givenFrame.index)
    givenFeatureMat = givenFeatureMat.tocsc()[remainderIndices,:]
    #then put information into sparse matrix
    osrmVars = ["total_distance","total_travel_time","number_of_steps"]
    osrmMat = sp.sparse.csc_matrix(givenFrame[osrmVars])
    givenFeatureMat = sp.sparse.hstack((givenFeatureMat,osrmMat))
    return givenFeatureMat, osrmCondition

def processTarget(givenFrame,osrmCondition,datasetType):
    #helper for getting our target export
    if (datasetType == "train"):
        givenFrame["logTripDuration"] = np.log(givenFrame["trip_duration"])
    else: #test frame
        givenFrame["logTripDuration"] = 0 #will alter this
        givenFrame["trip_duration"] = 1 #will alter this
    exportFrame = givenFrame[["id","trip_duration","logTripDuration"]]
    exportFrame = exportFrame[osrmCondition]
    return exportFrame

def process(givenFrame,holidayFrame,noiseComplaintFrame,weatherFrame,
            osrmFrame,datasetType):
    #main helper that processes our data
    assert(datasetType in ["train","test"])
    #process features
    featureMat, osrmCondition = processFeatures(givenFrame,holidayFrame,
                                                noiseComplaintFrame,
                                                weatherFrame,osrmFrame)
    #process target
    targetExport = processTarget(givenFrame,osrmCondition,datasetType)
    return featureMat, targetExport

#main process

if __name__ == "__main__":
    #load in raw datasets
    trainFrame = pd.read_csv("../data/raw/train.csv")
    testFrame = pd.read_csv("../data/raw/test.csv")
    #get additional datasets
    holidayFrame = pd.read_csv("../data/preprocessed/NYC_2016Holidays.csv",
                                sep = ";")
    trainNoiseComplaintFrame = pd.read_csv(
                            "../data/preprocessed/partynyc/train_parties.csv")
    testNoiseComplaintFrame = pd.read_csv(
                            "../data/preprocessed/partynyc/test_parties.csv")
    weatherFrame = pd.read_csv(
                "../data/preprocessed/weather_data_nyc_centralpark_2016.csv")
    trainFastestRouteFrame_p1 = pd.read_csv(
                    "../data/preprocessed/osrm/fastest_routes_train_part_1.csv")
    trainFastestRouteFrame_p2 = pd.read_csv(
                    "../data/preprocessed/osrm/fastest_routes_train_part_2.csv")
    trainFastestRouteFrame = pd.concat([trainFastestRouteFrame_p1,
                                        trainFastestRouteFrame_p2])
    testFastestRouteFrame = pd.read_csv(
                            "../data/preprocessed/osrm/fastest_routes_test.csv")
    #then process each to get feature frame
    trainMat, trainTargetExport = process(trainFrame,holidayFrame,
                                          trainNoiseComplaintFrame,
                                          weatherFrame,
                                          trainFastestRouteFrame,"train")
    testMat, testTargetExport = process(testFrame,holidayFrame,
                                        testNoiseComplaintFrame,
                                        weatherFrame,testFastestRouteFrame,
                                        "test")
    #then export
    exportDict = {"train":{},"test":{}}
    exportDict["train"]["featureMat"] = trainMat
    exportDict["train"]["target"] = trainTargetExport
    exportDict["test"]["featureMat"] = testMat
    exportDict["test"]["target"] = testTargetExport
    pkl.dump(exportDict,open("../data/processed/processedData.pkl","wb"))
