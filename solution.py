# Patient Clinic Proximity Finder Solution
import time
import numpy as np
import pandas as pd # for data manipulation and analysis
from sklearn.cluster import KMeans
import geocoder # for converting address string to geocode
from pypostalcode import PostalCodeDatabase # for getting geocode using the FSA
import requests # for sending http request to OSRM API
import json # for converting http response to json format
from tqdm import tqdm # for showing a progress bar


# open the configuration file
with open('config.json', 'r') as f:
    config = json.load(f)
# the number of time to retry a process for error handling
num_of_retries = config['num_of_retries']
# number of clusters for the kmeans algorithm
n_clusters = config['n_clusters']

# read the data with pandas
patients_df = pd.read_csv('data/patients.csv', index_col='ID')
clinics_df = pd.read_csv('data/clinics.csv', index_col='Clinic ID')
# rename the "Clinic Address" and  "Clinic City" columns to "Address" and "City"  respectively to match the patients dataset column names
clinics_df.rename(columns={"Clinic Address": "Address", "Clinic City": "City"}, inplace=True)


# the "Province" column stores the data in abbreviation format
# we'll create a function to convert the abbreviation to province name
def abrev_to_name(abrev):
    """ This function accepts province abbreviation and 
        returns the province name. """
    # create a dictionary maping all the province abbreviation 
    # to the province name
    can_province_names = {
      'AB': 'Alberta',
      'BC': 'British Columbia',
      'MB': 'Manitoba',
      'NB': 'New Brunswick',
      'NL': 'Newfoundland and Labrador',
      'NS': 'Nova Scotia',
      'NT': 'Northwest Territories',
      'NU': 'Nunavut',
      'ON': 'Ontario',
      'PE': 'Prince Edward Island',
      'QC': 'Quebec',
      'SK': 'Saskatchewan',
      'YT': 'Yukon'
    }
    # return the province name
    return can_province_names[abrev]


def geoArcgis(address):
    """ This function returns the geo location data of 
        a given string address using the "geocoder" module.
    """
    # Error Handling: using try except block to ensure continous 
    # execution in case there is a error for any particular row
    delay = 1
    for attempt in range(num_of_retries):
        try:
            # https://developers.arcgis.com/rest/geocode/api-reference/overview-world-geocoding-service.htm
            gcode = geocoder.arcgis(address)
            # return the response in a dictionary format
            return gcode.json
        except:
            if attempt < (num_of_retries - 1):
                # wait for some seconds before continuing
                time.sleep(delay) 
                # increase the delay wait time if the error persists
                delay *= 2
                continue
            else:
                return gcode.json


# we need to combine all the required column, 
# and pass each data point to "geoArcgis" function
def getGeoData(df):
    """ This function accepts a dataframe as an input.
        updates the dataframe with the extracted geo location data,
        and returns the new updated dataframe. """
    df = df.copy()
    # concatinating the columns required for getting the geocode
    df['Concat_Geo_Cols'] = df['Address'] + ", " + df['City'] + ", " + \
                            df['Province'].apply(abrev_to_name) + ", " + "Canada"
    # apply the "geoArcgis" function on all the rows
    # and store the returned data on a new column
    tqdm.pandas() # monitor and show the progress of the process
    df['details'] = df['Concat_Geo_Cols'].progress_apply(geoArcgis)
    # extract the longitude and latitude from the returned geo
    # location data and store them on seperate columns for easy access
    df['latitude'] = df['details'].apply(lambda x: x['lat'] if x else np.NaN)
    df['longitude'] = df['details'].apply(lambda x: x['lng'] if x else np.NaN)
    # return the updated dataframe with the geodata
    return df
    

def postalCodeDatabase(fsa):
    """ 
        This function finds the geo location(logitude and latitude )
        using the patient or clinic FSA
    """
    pcdb = PostalCodeDatabase()
    location = pcdb[fsa]    
    return {'lng': location.longitude, 'lat': location.latitude}


def cluster_geocodes():
    """ 
        This function clusters the clinic dataset 
        based on it logitude and latitude using "Kmeans cluster algorithm"
    """
    df = clinics_df[['longitude', 'latitude']].copy()
    #dummy = pd.get_dummies(df['FSA'])
    #df = pd.concat([df, dummy], axis=1)
    #df.drop('FSA', inplace=True, axis=1)
    
    # define model
    kmeans = KMeans(n_clusters=n_clusters)
    clinics_df["Cluster"] = kmeans.fit_predict(df)
    clinics_df["Cluster"] = clinics_df["Cluster"].astype("category")

    return kmeans

def getDrivingDistance(clinic_gcode, patient_gcode):
    """ This function gets the distance between a patient and a
        a clinic using the "router.project-osrm.org" API 
        when provided with the logitude and latitude of the two locations """
    # Error Handling: using try except block to ensure continous 
    # execution in case there is a error for any particular row
    delay = 1
    for attempt in range(num_of_retries):
        try:
            # call the OSMR API
            r = requests.get(f"http://router.project-osrm.org/route/v1/car/{patient_gcode['lng']},{patient_gcode['lat']};{clinic_gcode['lng']},{clinic_gcode['lat']}?overview=false""", timeout=5)
            if r.status_code == 200:
                routes = json.loads(r.content)
                route_ = routes.get("routes")[0]
                # return distance in kilometer by dividing the meter value with 1000
                return route_['distance']/1000
            else:
                return np.NaN
        except requests.exceptions.RequestException as e:
            if attempt < (num_of_retries - 1):
                print('Error:', e)
                # wait for some seconds before continuing
                time.sleep(delay) 
                # increase the delay wait time if the error persists
                delay *= 2
                continue
            else:
                return np.NaN
        except TypeError:
            return np.NaN

def findNearestClinic(patient_gcode, kmeans=None):
    """Check if clusters were created, and find 
        the nearest clinic based on its cluster.
        If not, search the whole clinic dataset for 
        nearest clinic.
    """
    if kmeans:
        # using the patients geocode, get the cluster it belongs to
        cluster = kmeans.predict([[patient_gcode['lng'], patient_gcode['lat']]])[0]
        clinics_data = clinics_df[clinics_df['Cluster'] == cluster].copy()
    else:
        clinics_data = clinics_df.copy()
    # apply the "getDrivingDistance" function on each row of the clinic data with a particular patient
    clinics_data['distance'] = clinics_data['details'].apply(getDrivingDistance, args=(patient_gcode,))
    # return the minimum distance and the Id of the clinic with the minimum distance
    try:
        return {'clinic_id':clinics_data['distance'].idxmin(), 'distance':clinics_data['distance'].min()}
    except:
        return None
                
def getOutput(df, kmeans=None):
    """ This function tries to arrange the program output
        as specified in the challenge description. """
    df = df.copy()
    # apply the "findNearestClinic" function to the every patient in the dataset
    tqdm.pandas() # monitor and show the progress of the process
    df['NearestClinic'] = df.details.progress_apply(findNearestClinic, args=(kmeans,))
    # separate the clinic Id and the distance on different columns
    df['Nearest_Clinic_ID'] = df['NearestClinic'].apply(lambda x: x['clinic_id'] if x else None)
    df['Clinic_Distance'] = df['NearestClinic'].apply(lambda x: x['distance'] if x else None)
    # columns used to find the geocode of the patient
    df['Pat_Geo_Cols'] = 'Address, Province, City'
    # calculated lat/long coordinates of the patient
    df['Pat_Geocode'] = df['longitude'].astype(str) + ', ' + df['latitude'].astype(str)
    # rename Address, Postal and FSA columns to Pat_Address, Pat_Postal_Code and Pat_FSA 
    # respectively to match the output requirement
    df.rename(columns={"Address": "Pat_Address", "Postal Code": "Pat_Postal_Code", "FSA": "Pat_FSA"}, inplace=True)
    # columns used to find the geocode of the clinic
    df['Clinic_Geo_Cols'] = 'Address, Province, City' 
    # get the geocode, address, postal code, FSA of the nearest clinic using the "Nearest_Clinic_ID"
    df['Clinic_Geocode'] = df['Nearest_Clinic_ID'][df['Nearest_Clinic_ID'].notnull()].apply(
                            lambda x: f"{clinics_df.loc[x, 'longitude']}" + \
                            f"{clinics_df.loc[x, 'latitude']}" if x else None)
    df['Clinic_Address'] = df['Nearest_Clinic_ID'][df['Nearest_Clinic_ID'].notnull()].apply(
                            lambda x: clinics_df.loc[x, 'Address'] if x else None)
    df['Clinic_Postal Code'] = df['Nearest_Clinic_ID'][df['Nearest_Clinic_ID'].notnull()].apply(
                            lambda x: clinics_df.loc[x, 'Postal Code'] if x else None)
    df['Clinic_FSA'] = df['Nearest_Clinic_ID'][df['Nearest_Clinic_ID'].notnull()].apply(
                            lambda x: clinics_df.loc[x, 'FSA'] if x else None)
    
    
    # drop the remaining columns not needed for the output
    df.drop(['City','Province','Concat_Geo_Cols','details','latitude','longitude','NearestClinic'], axis=1, inplace=True)
    # making sure all the column names corresponds to the required output
    df.index.names = ['Patient_ID']
    return df[['Pat_Geo_Cols', 'Pat_Geocode', 'Pat_Address', 'Pat_Postal_Code', 'Pat_FSA', 
               'Nearest_Clinic_ID', 'Clinic_Geo_Cols', 'Clinic_Geocode', 'Clinic_Address', 
               'Clinic_Postal Code', 'Clinic_FSA', 'Clinic_Distance', 'Nearest_Clinic_ID']]



if __name__ == "__main__":
    # get the geocode/data for patients dataset
    print("Getting patients geocode...")
    patients_df = getGeoData(patients_df)
    print("Done!")


    # get the geocode/data for clinics dataset
    print("Getting clinics geocode...")
    clinics_df = getGeoData(clinics_df)
    print("Done!")

    """ 
    Incase if the geocode a patient is not found using the
    patient's address in conjuction with the geocoder module, 
    the block of code below finds the the geocode using the "FSA"
    column with PostalCodeDatabase class from pypostalcode module.
    """
    if len(patients_df.details.isna()):
        patients_df.loc[patients_df['details'].isnull(), 'details'] = patients_df[
                    patients_df['details'].isnull()]['FSA'].apply(postalCodeDatabase)
        patients_df.loc[patients_df['latitude'].isnull(), 'latitude'] = patients_df[
                    patients_df['latitude'].isnull()]['details'].apply(lambda x: x['lat'])
        patients_df.loc[patients_df['longitude'].isnull(), 'longitude'] = patients_df[
                    patients_df['longitude'].isnull()]['details'].apply(lambda x: x['lng'])
    

    """ 
    Incase if the geocode a clinic is not found using the
    clinic's address in conjuction with the geocoder module, 
    the block of code below finds the the geocode using the "FSA"
    column with PostalCodeDatabase class from pypostalcode module.
    """
    if len(clinics_df.details.isna()):
        clinics_df.loc[clinics_df['details'].isnull(), 'details'] = clinics_df[
                    clinics_df['details'].isnull()]['FSA'].apply(postalCodeDatabase)
        clinics_df.loc[clinics_df['latitude'].isnull(), 'latitude'] = clinics_df[
                    clinics_df['latitude'].isnull()]['details'].apply(lambda x: x['lat'])
        clinics_df.loc[clinics_df['longitude'].isnull(), 'longitude'] = clinics_df[
                    clinics_df['longitude'].isnull()]['details'].apply(lambda x: x['lng'])
    
    # to enable the program search for a patient's nearest clinic 
    # based on it's cluster instead of searching the whole dataset
    kmeans = cluster_geocodes()

    # display and save the expected output as a csv file
    print("Getting output data...")
    output_df = getOutput(patients_df, kmeans)
    print("Done!")
    print(output_df)
    # save output as output.csv
    output_df.to_csv('output.csv')
