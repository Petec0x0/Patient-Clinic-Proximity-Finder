## Patient Clinic Proximity Finder

### Overview
The reqiured Python version for this program is Python 3.9.2
- The `solution.py` contains the main code of the program.
- `requirements.txt` contains the packages that needs to be installed before running the programm.
- `config.json` contains the program configurations.
- `data/clinics.csv` contains clinics dataset.
- `data/patients.csv` contains patients dataset.
- `output.csv` output file based on the challenge dsecription.

Installing packages:
use `$ pip install -r requirements.txt` OR `$ pip3 install -r requirements.txt` to install the required packages for the program.

Running the program:
`$ python solution.py`
OR
`$ python3 solution.py`


#### Approach used in the solution
The goal of the program is to matches patients to their nearest clinic using the provided patients and clinic dataset.
The program uses 'pandas' library for reading and manipulating the datasets.
For getting the patients and clinics geocode(logitude/latitude), `arcgis()` method for `geocoder` package was used in this case, it accepts a natural language string address in string format and returns the geolocation data. The sting format address passed
into the `arcgis()` method is a combination from "Address, City and Province" column plus country name "Canada".

The "Province" column in the dataset stores the province in abbreviation format. It is first coverted to the province names 
before being used for getting the geocode.

In the `geoArcgis()` function of the program, a try catch block is used to mitigate the program from crashing in case if the API request to arcgis service fails because of some network issue. The program retries the request N number of times(specified in the 'config.json') and then continues to the next row if the error persists.

In case the geocode of any address is not found using the `geoArcgis()` function, the `postalCodeDatabase(fsa)` function is 
applied to the rows with missing geocode by taking the "FSA" as input and returning the gecode.

For calculating the commute distance, a http request is sent to http://router.project-osrm.org/ API which takes the logitude and latitude of two locations as parameters and returns the distance in meters. The distance is converted to kilometers by dividing the meter distance with 1000.

The program clusters the dataset into N clusters (this depends on the number of datasets used and it is specified in the 'config.json' as n_clusters) using the Kmeans clustering algorithm. This is to enable the program search for a patient's nearest clinic based on it's cluster instead of searching the whole dataset. The clustering algorithm uses the "logitude and latitude" as features for the model.

When the nearest clinic is found, the `getOutput()` function arranges the required output as specified in the challenge description.