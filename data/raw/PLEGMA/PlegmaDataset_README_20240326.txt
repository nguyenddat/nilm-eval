Plegma Dataset: A comprehensive appliance-level and aggregate electricity demand, with environmental and demographic data from 13 domestic houses in Greece

INFORMATION
Collection of this dataset was supported by the Plegma Labs, National Technical University of Athens and University of Strathclyde via the the GECKO Marie
Skłodowska-Curie grant agreement No 955422. The dataset includes data from 13 households from Greece over the period of one year starting July 2022.

FOLDER STRUCTURE
Plegma Dataset contains electric, environmental and demographic data of households that participate
in the study. Plegma Dataset is also provided in both raw and preprocessed version in order to give
the chance to the research community to either use our or create their own preprocessing pipeline.
The 'sociodemographic-building characteristics-appliances usage', external environmental data and
appliances metadata are only presented in the processed version of the dataset.
The finalised folder structure is this:

Plegma Dataset |
	       |
               -- Raw Dataset|
	       |	     |
               |             -- House_1 |
	       |                        |
	       |                        --Raw_collected_data (folder with CSV files / month-year)
               |                        | 
               |                        --Metadata_raw_H1 (CSV file / month-year)
               |              .
               |              . 
               |              .
               |
               |
               -- CleanDataset|
                              |
                              --House_1 |
                                        --Electric_data (CSV file / month-year + appliances metadata wattage)
                                        |
                                        --Environmental_data (CSV file /month-year)
                                        |
                                        -- Sociodemographic/
                                          Building Characteristics/
                                          Time of use of Appliances 
                                        |
                                        -- Metadata_H1
          
                                               

GRANULARITY
Houses are labelled from 1 to 13. Each house is equipped with up to 9 plug-level power sensors, a current clamp for the household aggregate and an environmental sensor. For the appliance-level measurements only active power in Watts is collected at 10 second interval. For the aggregate consumption active power, voltage and current measurments are 
included in the same granularity of 10 seconds. The environmental sensor monitors
internal humidity (%) and temperature (°C) in 15 min granularity. The hourly external
weather conditions (temperature & humidity) are also provided, downsampled to 15 min to match
the monitored internal ones.

FILE FORMAT and NAMING CONVENTION
The file format is CSV (comma separated values) and is organised out as follows;
DateTime (DAY/MONTH/YEAR  00:00:00 AM), Voltage [Volt], Current [Ampere], Active Power [Watts], Appliance 1, ..., Appliance N.
The naming convention of the electric dataset is shown below. In houses that contain more than one 
appliance of the same type, the appliance name is followed by an underscore and 
an increasing number (i.e., ac_1).

Aggregate smart meter (V): Voltage
Aggregate smart meter (A): Current
Aggregate smart meter (W): P_agg
Air Condition            : ac
Boiler                   : boiler
Dishwasher               : dishwasher
Fridge                   : fridge
Washing Machine          : washing_machine

The last column denoted 'Issues', points to the rows where the sum of the monitored appliances
is larger than the aggregate. This is a common flaw in many datasets and the main reason is the
equipment error.
Each CSV file includes measurements up to a month so it is able to display it in Excel. However, it is 
recommended to open the files using a program such as Matlab or other tabular data processing applications.

 
MISSING DATA
Throughout the study, there were several instances of data unavailability caused by various factors, such as household internet disruptions, hardware malfunctions, and network routing problems.

DATA CLEANING
The data has been cleaned to forward fill small gaps of NaN values and remove values which exceed the wattage range of the monitored appliances. In the electric dataset, where the granularity is set to 10sec, gaps more than 30sec remained as NaN while gaps shorter than 30s were forward filled. For the environmental data small gaps (<=60 min) were forward filled while larger gaps were remained an NaN.

APPLIANCES METADATA
The electric folder of the processed dataset includes the appliances metadata which is a
CSV file containing wattage information such as wattage, threshold, min_on, min_off. wattage expresses the active power monitored from each appliance, threshold expresses the minimum wattage for an appliance to consider as on, min_on is the minimum time for an appliance to be considered as on, and min_off expresses the minimum time that an appliance should have a wattage below the threshold to considered as off.
