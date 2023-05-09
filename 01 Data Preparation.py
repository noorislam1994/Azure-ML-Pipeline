# Databricks notebook source
# MAGIC %md # Prepare flight delay data

# COMMAND ----------

# MAGIC %md To start, let's import the Python libraries and modules we will use in this notebook.

# COMMAND ----------

import pprint, datetime
from pyspark.sql.types import *
from pyspark.sql.functions import unix_timestamp
import math
from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %md First, let's execute the below command to make sure all three tables were created.
# MAGIC You should see an output like the following:
# MAGIC
# MAGIC | database | tableName | isTemporary |
# MAGIC | --- | --- | --- |
# MAGIC | default | airport_code_loca... | false |
# MAGIC | default | flight_delays_wit... | false |
# MAGIC | default | flight_weather_wi... | false |

# COMMAND ----------

spark.sql("show tables").show()

# COMMAND ----------

# MAGIC %md Now execute a SQL query using the `%sql` magic to select all columns from flight_delays_with_airport_codes. By default, only the first 1,000 rows will be returned.

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from flight_delays_with_airport_codes

# COMMAND ----------

# MAGIC %md Now let's see how many rows there are in the dataset.

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from flight_delays_with_airport_codes

# COMMAND ----------

# MAGIC %md Based on the `count` result, you can see that the dataset has a total of 2,719,418 rows (also referred to as examples in Machine Learning literature). Looking at the table output from the previous query, you can see that the dataset contains 20 columns (also referred to as features).

# COMMAND ----------

# MAGIC %md Because all 20 columns are displayed, you can scroll the grid horizontally. Scroll until you see the **DepDel15** column. This column displays a 1 when the flight was delayed at least 15 minutes and 0 if there was no such delay. In the model you will construct, you will try to predict the value of this column for future data.
# MAGIC
# MAGIC Let's execute another query that shows us how many rows do not have a value in the DepDel15 column.

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from flight_delays_with_airport_codes where DepDel15 is null

# COMMAND ----------

# MAGIC %md Notice that the `count` result is 27444. This means that 27,444 rows do not have a value in this column. Since this value is very important to our model, we will need to eliminate any rows that do not have a value for this column.

# COMMAND ----------

# MAGIC %md Next, scroll over to the **CRSDepTime** column within the table view above. Our model will approximate departure times to the nearest hour, but departure time is captured as an integer. For example, 8:37 am is captured as 837. Therefore, we will need to process the CRSDepTime column, and round it down to the nearest hour. To perform this rounding will require two steps, first you will need to divide the value by 100 (so that 837 becomes 8.37). Second, you will round this value down to the nearest hour (so that 8.37 becomes 8).

# COMMAND ----------

# MAGIC %md Finally, we do not need all 20 columns present in the flight_delays_with_airport_codes dataset, so we will pare down the columns, or features, in the dataset to the 12 we do need.

# COMMAND ----------

# MAGIC %md Using `%sql` magic allows us view and visualize the data, but for working with the data in our tables, we want to take advantage of the rich optimizations provided by DataFrames. Let's execute the same query using Spark SQL, this time saving the query to a DataFrame.

# COMMAND ----------

dfFlightDelays = spark.sql("select * from flight_delays_with_airport_codes")

# COMMAND ----------

# MAGIC %md Let's print the schema for the DataFrame.

# COMMAND ----------

pprint.pprint(dfFlightDelays.dtypes)

# COMMAND ----------

# MAGIC %md Notice that the DepDel15 and CRSDepTime columns are both `string` data types. Both of these features need to be numeric, according to their descriptions above. We will cast these columns to their required data types next.

# COMMAND ----------

# MAGIC %md ## Perform data munging

# COMMAND ----------

# MAGIC %md To perform our data munging, we have multiple options, but in this case, we’ve chosen to take advantage of some useful features of R to perform the following tasks:
# MAGIC
# MAGIC *	Remove rows with missing values
# MAGIC *	Generate a new column, named “CRSDepHour,” which contains the rounded down value from CRSDepTime
# MAGIC *	Pare down columns to only those needed for our model

# COMMAND ----------

# MAGIC %md SparkR is an R package that provides a light-weight frontend to use Apache Spark from R. To use SparkR we will call `library(SparkR)` within a cell that uses the `%r` magic, which denotes the language to use for the cell. The SparkR session is already configured, and all SparkR functions will talk to your attached cluster using the existing session.

# COMMAND ----------

# MAGIC %r
# MAGIC library(SparkR)
# MAGIC
# MAGIC # Select only the columns we need, casting CRSDepTime as long and DepDel15 as int, into a new DataFrame
# MAGIC dfflights <- sql("SELECT OriginAirportCode, OriginLatitude, OriginLongitude, Month, DayofMonth, cast(CRSDepTime as long) CRSDepTime, DayOfWeek, Carrier, DestAirportCode, DestLatitude, DestLongitude, cast(DepDel15 as int) DepDel15 from flight_delays_with_airport_codes")
# MAGIC
# MAGIC # Delete rows containing missing values
# MAGIC dfflights <- na.omit(dfflights)
# MAGIC
# MAGIC # Round departure times down to the nearest hour, and export the result as a new column named "CRSDepHour"
# MAGIC dfflights$CRSDepHour <- floor(dfflights$CRSDepTime / 100)
# MAGIC
# MAGIC # Trim the columns to only those we will use for the predictive model
# MAGIC dfflightsClean = dfflights[, c("OriginAirportCode","OriginLatitude", "OriginLongitude", "Month", "DayofMonth", "CRSDepHour", "DayOfWeek", "Carrier", "DestAirportCode", "DestLatitude", "DestLongitude", "DepDel15")]
# MAGIC
# MAGIC createOrReplaceTempView(dfflightsClean, "flight_delays_view")

# COMMAND ----------

# MAGIC %md Now let's take a look at the resulting data. Take note of the **CRSDepHour** column that we created, as well as the number of columns we now have (12). Verify that the new CRSDepHour column contains the rounded hour values from our CRSDepTime column.

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from flight_delays_view

# COMMAND ----------

# MAGIC %md Now verify that the rows with missing data for the **DepDel15** column have been removed.

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from flight_delays_view

# COMMAND ----------

# MAGIC %md You should see a count of **2,691,974**. This is equal to the original 2,719,418 rows minus the 27,444 rows with missing data in the DepDel15 column.
# MAGIC
# MAGIC Now save the contents of the temporary view into a new DataFrame.

# COMMAND ----------

dfFlightDelays_Clean = spark.sql("select * from flight_delays_view")

# COMMAND ----------

# MAGIC %md ## Export the prepared data to persistent a global table

# COMMAND ----------

# MAGIC %md There are two types of tables in Databricks. 
# MAGIC
# MAGIC * Global tables, which are accessible across all clusters
# MAGIC * Local tables, which are available only within one cluster
# MAGIC
# MAGIC To create a global table, you use the `saveAsTable()` method. To create a local table, you would use either the `createOrReplaceTempView()` or `registerTempTable()` method.
# MAGIC
# MAGIC The `flight_delays_view` table was created as a local table using `createOrReplaceTempView`, and is therefore temporary. Local tables are tied to the Spark/SparkSQL Context that was used to create their associated DataFrame. When you shut down the SparkSession that is associated with the cluster (such as shutting down the cluster) then local, temporary tables will disappear. If we want our cleansed data to remain permanently, we should create a global table. 
# MAGIC
# MAGIC Run the following to copy the data from the source location into a global table named `flight_delays_clean`.

# COMMAND ----------

dfFlightDelays_Clean.write.mode("overwrite").saveAsTable("flight_delays_clean")

# COMMAND ----------

# MAGIC %md # Prepare the weather data

# COMMAND ----------

# MAGIC %md To begin, take a look at the `flight_weather_with_airport_code` data that was imported to get a sense of the data we will be working with.

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from flight_weather_with_airport_code

# COMMAND ----------

# MAGIC %md Next, count the number of records so we know how many rows we are working with.

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from flight_weather_with_airport_code

# COMMAND ----------

# MAGIC %md Observe that this data set has 406,516 rows and 29 columns. For this model, we are going to focus on predicting delays using WindSpeed (in MPH), SeaLevelPressure (in inches of Hg), and HourlyPrecip (in inches). We will focus on preparing the data for those features.

# COMMAND ----------

# MAGIC %md Let's start out by taking a look at the **WindSpeed** column. You may scroll through the values in the table above, but reviewing just the distinct values will be faster.

# COMMAND ----------

# MAGIC %sql
# MAGIC select distinct WindSpeed from flight_weather_with_airport_code

# COMMAND ----------

# MAGIC %md Try clicking on the **WindSpeed** column header to sort the list by ascending and then by descending order. Observe that the values are all numbers, with the exception of some having `null` values and a string value of `M` for Missing. We will need to ensure that we remove any missing values and convert WindSpeed to its proper type as a numeric feature.

# COMMAND ----------

# MAGIC %md Next, let's take a look at the **SeaLevelPressure** column in the same way, by listing its distinct values.

# COMMAND ----------

# MAGIC %sql
# MAGIC select distinct SeaLevelPressure from flight_weather_with_airport_code

# COMMAND ----------

# MAGIC %md Like you did before, click on the **SeaLevelPressure** column header to sort the values in ascending and then descending order. Observe that many of the features are of a numeric value (e.g., 29.96, 30.01, etc.), but some contain the string value of M for Missing. We will need to replace this value of "M" with a suitable numeric value so that we can convert this feature to be a numeric feature.

# COMMAND ----------

# MAGIC %md Finally, let's observe the **HourlyPrecip** feature by selecting its distinct values.

# COMMAND ----------

# MAGIC %sql
# MAGIC select distinct HourlyPrecip from flight_weather_with_airport_code

# COMMAND ----------

# MAGIC %md Click on the column header to sort the list and ascending and then descending order. Observe that this column contains mostly numeric values, but also `null` values and values with `T` (for Trace amount of rain). We need to replace T with a suitable numeric value and convert this to a numeric feature.

# COMMAND ----------

# MAGIC %md ## Clean up weather data

# COMMAND ----------

# MAGIC %md To preform our data cleanup, we will execute a Python script, in which we will perform the following tasks:
# MAGIC
# MAGIC * WindSpeed: Replace missing values with 0.0, and “M” values with 0.005
# MAGIC * HourlyPrecip: Replace missing values with 0.0, and “T” values with 0.005
# MAGIC * SeaLevelPressure: Replace “M” values with 29.92 (the average pressure)
# MAGIC * Convert WindSpeed, HourlyPrecip, and SeaLevelPressure to numeric columns
# MAGIC * Round “Time” column down to the nearest hour, and add value to a new column named “Hour”
# MAGIC * Eliminate unneeded columns from the dataset

# COMMAND ----------

# MAGIC %md Let's begin by creating a new DataFrame from the table. While we're at it, we'll pare down the number of columns to just the ones we need (AirportCode, Month, Day, Time, WindSpeed, SeaLevelPressure, HourlyPrecip).

# COMMAND ----------

dfWeather = spark.sql("select AirportCode, cast(Month as int) Month, cast(Day as int) Day, cast(Time as int) Time, WindSpeed, SeaLevelPressure, HourlyPrecip from flight_weather_with_airport_code")

# COMMAND ----------

dfWeather.show()

# COMMAND ----------

# MAGIC %md Review the schema of the dfWeather DataFrame

# COMMAND ----------

pprint.pprint(dfWeather.dtypes)

# COMMAND ----------

# Round Time down to the next hour, since that is the hour for which we want to use flight data. Then, add the rounded Time to a new column named "Hour", and append that column to the dfWeather DataFrame.
df = dfWeather.withColumn('Hour', F.floor(dfWeather['Time']/100))

# Replace any missing HourlyPrecip and WindSpeed values with 0.0
df = df.fillna('0.0', subset=['HourlyPrecip', 'WindSpeed'])

# Replace any WindSpeed values of "M" with 0.005
df = df.replace('M', '0.005', 'WindSpeed')

# Replace any SeaLevelPressure values of "M" with 29.92 (the average pressure)
df = df.replace('M', '29.92', 'SeaLevelPressure')

# Replace any HourlyPrecip values of "T" (trace) with 0.005
df = df.replace('T', '0.005', 'HourlyPrecip')

# Be sure to convert WindSpeed, SeaLevelPressure, and HourlyPrecip columns to float
# Define a new DataFrame that includes just the columns being used by the model, including the new Hour feature
dfWeather_Clean = df.select('AirportCode', 'Month', 'Day', 'Hour', df['WindSpeed'].cast('float'), df['SeaLevelPressure'].cast('float'), df['HourlyPrecip'].cast('float'))


# COMMAND ----------

# MAGIC %md Now let's take a look at the new `dfWeather_Clean` DataFrame.

# COMMAND ----------

display(dfWeather_Clean)

# COMMAND ----------

# MAGIC %md Observe that the new DataFrame only has 7 columns. Also, the WindSpeed, SeaLevelPressure, and HourlyPrecip fields are all numeric and contain no missing values. To ensure they are indeed numeric, we can take a look at the DataFrame's schema.

# COMMAND ----------

pprint.pprint(dfWeather_Clean.dtypes)

# COMMAND ----------

# MAGIC %md Now let's persist the cleaned weather data to a persistent global table.

# COMMAND ----------

dfWeather_Clean.write.mode("overwrite").saveAsTable("flight_weather_clean")

# COMMAND ----------

dfWeather_Clean.select("*").count()

# COMMAND ----------

# MAGIC %md # Join the Flight and Weather datasets

# COMMAND ----------

# MAGIC %md With both datasets ready, we want to join them together so that we can associate historical flight delays with the weather data at departure time.

# COMMAND ----------

dfFlightDelaysWithWeather = spark.sql("SELECT d.OriginAirportCode, \
                 d.Month, d.DayofMonth, d.CRSDepHour, d.DayOfWeek, \
                 d.Carrier, d.DestAirportCode, d.DepDel15, w.WindSpeed, \
                 w.SeaLevelPressure, w.HourlyPrecip \
                 FROM flight_delays_clean d \
                 INNER JOIN flight_weather_clean w ON \
                 d.OriginAirportCode = w.AirportCode AND \
                 d.Month = w.Month AND \
                 d.DayofMonth = w.Day AND \
                 d.CRSDepHour = w.Hour")

# COMMAND ----------

# MAGIC %md Now let's take a look at the combined data.

# COMMAND ----------

display(dfFlightDelaysWithWeather)

# COMMAND ----------

# MAGIC %md Write the combined dataset to a new persistent global table.

# COMMAND ----------

dfFlightDelaysWithWeather.write.mode("overwrite").saveAsTable("flight_delays_with_weather")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next step
# MAGIC
# MAGIC Continue to the next notebook, [02 Train and Evaluate Models]($./02%20Train%20and%20Evaluate%20Models).