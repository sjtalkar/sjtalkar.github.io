## DP - 100
### Spark RDD SQL Intro to Parquet (columnar compression metadata)

#### What's the need being served
 **Big data - which is?**
 - Three Vs : Volume, Velocity, Variety 

 **Resilient Distributed Datasets**
 - Fault tolerant (recomputation in case of failure)
 - Parallelization and partitioning (Split into nodes within a cluster and also CPUs within a node)
 
 
As you may know, Spark is a Distributed computing environment. The unit of distribution is a Spark Cluster. Every Cluster has a Driver and one or more executors. Work submitted to the Cluster is split into as many independent Jobs as needed. This is how work is distributed across the Cluster's nodes. Jobs are further subdivided into tasks.

* The first level of parallelization is the Executor - a Java virtual machine running on a node, typically, one instance per node.
* The second level of parallelization is the Slot - the number of which is determined by the number of cores and CPUs of each node.

##### Transformations and Actions

![First Picture](https://github.com/sjtalkar/DP-100AzureSupervisedUnsupervisedDatabricksAndSpark/blob/main/Pictures%20for%20Readme/Picture1.png)

It's not until we induce an action that a job is triggered and the data is processed. **Lazy** being the key word

Let's take the task of reading a file.
Inferring schema involves:

- Knowing column names
- Knowing the type of the data in the column
- Knowing if the column contains null

When loading data for instance, using Parquet file saves on Inferring Schema. Number of jobs when reading a Parquet files is typically 0 because of the meta data available intrinsically in Parquet files. For that matter while using JSON files saves on one job (column names and structure  are in the keys, but data types and nulls need to be inferred). 
When the data has to be physically touched - that's when an Executor needs to roll up it's sleeve and get to work - to accomplish a Job.

For operations such as select, withColumns, map and other transforations seen above, things are kept on hold until an action demanding data is called such as show, count, collect (DANGER!!) and save.

> An aside on Parquet
>
>About Parquet => (with METADATA Files!)
>
>* Free & Open Source.
>* A Column-Oriented data store
>* Increased query performance over row-based data stores.
>* Provides efficient data compression.
>* Designed for performance on large data sets.
>* Supports limited schema evolution.
>* Is a splittable "file format".



#### Let's DIVE in : Setting up the spark session to get things going
The Azure Databricks environment provides us with a Spark session - the object is named "spark". 
Spark contexts can be created within a Spark session to work with Resilient Distributed Datasets. To read and load data we use the Spark session object. In a notebook you can create a Spark session with:

```python
from pyspark.sql import SparkSession
spark = SparkSession \
    .builder \
    .master("local[*]") \
    .appName('My First Spark application') \
    .getOrCreate() 

sc = spark.sparkContext
```


#### All that you can load
Now a Parquet data load does not benefit from it, but knowing your schema as in column headers names type of column and nullable in advance, reduces the number of jobs since you do not have to inferSchema. Parquet files come with the metadata that helps avoid inferring schema but CSVs and JSON loads can benefit from the schema definition shown below.

The read structure is similar to that in Pandas in that you can specify delimitiers and if there is header and so on.

For instance to read an Inside AirBnB csv file that has way too many columns to define schema on (not that it cannot be laboriously performed), we can set inferSchema to True (this unfortunately ensures a job will be created). If the records can possible broken up by a newline character, set multiline to True.

[Spark Multiline](https://sparkbyexamples.com/spark/spark-read-multiline-multiple-line-csv-file/)

You will also notice escape set to the double quotes character- this is to preserve the qutotes in string. The difference in amenities column with and without double quotes can be seen as:
This string with escape = '"' set

{TV,"Cable TV",Internet,Wifi,Kitchen,"Pets live on this property",Dog(s),Heating,"Family/kid friendly",Washer,Dryer,"Smoke detector","Carbon monoxide detector","First aid kit",Essentials,Shampoo,"24-hour check-in",Hangers,"Hair dryer",Iron,"Laptop friendly workspace","Self check-in",Keypad,"Private entrance","Pack ’n Play/travel crib","Room-darkening shades"}

Turns to this if double quotes is not escaped:

"{TV,""Cable TV""

In Databricks, the display function prettifies the Dataframe so that you can see the data in rows muc better than if you use Spark's show

!["Display of Table"](https://github.com/sjtalkar/DP-100AzureSupervisedUnsupervisedDatabricksAndSpark/blob/main/Pictures%20for%20Readme/DBDisplay.JPG)

```python
filePath = "dbfs:/mnt/training/airbnb/sf-listings/sf-listings-2019-03-06.csv"
rawDF = spark.read.csv(filePath, header=True, inferSchema=True, multiLine=True, escape='"')
display(rawDF)
```


NOTE all types such as StringType, IntegerType and so on have to be imported.
```python
from pyspark.sql.types import *

parquetSchema = StructType([
  StructField("project", StringType(), False),
  StructField("article", StringType(), False),
  StructField("requests", IntegerType(), False),
  StructField("bytes_served", IntegerType(), False)
  
])
# Use the schema defined above
df = (spark
  .read           # The DataFrameReader
  .schema(parquetSchema)           # Use the specified schema
  .parquet(path)                   # Creates a DataFrame from PARQUET after reading in the file
)

# The below finds the number of distinct articles in the files 
totalArticles = df.select('article').distinct().count() # Identify the total number of records remaining.
print("Distinct Articles: {0:,}".format(totalArticles))


# Read file from Github

%sh curl -O "https://raw.githubusercontent.com/sjtalkar/SeriallyBuildDashboard/main/data/listings_1.csv"

# check out where the file was stored

%fs ls "file:/databricks/driver"


# or dbutils.fs.ls("file:/databricks/driver")

# Read in CSV to DataFrame using above path

path = 'file:/databricks/driver/listings_1.csv'

# load data using sqlContext

airbnb_df  = spark.read.csv(path, header=True, inferSchema=True, multiLine=True, escape='"')

# display in table format
display(airbnb_df)


# Create a function to get first letter of host name ( I know a trivial task)
def firstInitialFunction(name):
  return name[0]

firstInitialFunction("Jane")

To create a udf from the function that can be applied on the dataframe column
firstInitialUDF = udf(firstInitialFunction)

#Employ the UDF
from pyspark.sql.functions import col
display(airbnb_df.select(firstInitialUDF(col("host_name"))))
```

#### To create a registered UDF from the function that can be used within a SQL query

```python
from pyspark.sql.types import *
spark.udf.register("firstInitialRegisteredUDF", firstInitialFunction,  StringType())

#Employ the registered UDF 
#NOTE: convert the Dataframe into a view so that it can be used in the query!!!

airbnb_df.createOrReplaceTempView("airbnbDF")

%sql
select distinct firstInitialRegisteredUDF(host_name) 
from airbnbDF
```

Since UDFs can be time consuming 
- they need to be serialized for the executor
- and applied row by row, use pre-defined functions or vectorized UDFs
The below is a UDF defined by a "decorator" pandas_udf is a vectorized UDF versus just udf which is a line by line udf

```python
%python
from pyspark.sql.functions import pandas_udf

# We have a string input/output
@pandas_udf("string")
def vectorizedUDF(name):
  return name.str[0]


# Text file can be read with read.csv

# Define the schema to reduce jobs
from pyspark.sql.types import *

textSchema = StructType([
  StructField("firstName", StringType(), False),
  StructField("middleName", StringType(), False),
  StructField("lastName", StringType(), False),
  StructField("gender", StringType(), False),
  StructField("birthDate", TimestampType(), False),
  StructField("salary", FloatType(), False),
  StructField("ssn", StringType(), False)
    
])
exercise_df = (
               spark
              .read
              .schema(textSchema)
              .option("sep", ":")
              .option("header", "true") 
              .csv(sourceFile)
               
)
```

All that you can avoid
Aim : Reduce the number of jobs that are spawned and are to be executed.

All that you can register
UDFs - registering for usage in an SQL query
Registering a dataframe as a view again so that it can be used in the query
### create a temporary view from the resulting DataFrame

```python
parquetDF.createOrReplaceTempView("parquet_table")
Once registered the function or the view can be used in the SQL command
%sql
select * from parquet_table order by requests desc limit(5)



from pyspark.sql.types import *

textSchema = StructType([
  StructField("firstName", StringType(), False),
  StructField("middleName", StringType(), False),
  StructField("lastName", StringType(), False),
  StructField("gender", StringType(), False),
  StructField("birthDate", TimestampType(), False),
  StructField("salary", FloatType(), False),
  StructField("ssn", StringType(), False)
    
])


exercise_df = (
               spark
              .read
              .schema(textSchema)
              .option("sep", ":")
              .option("header", "true") 
              .csv(sourceFile)
               
)

def capitalizeString(textString):
  return textString.capitalize()
firstCapitalizeUDF = udf(capitalizeString)

def formatSSN(ssnText):
  #Strip off hyphens if any to bring all SSNs to the same format and then insert them
  ssnText = ssnText.replace('-', '')
  return ssnText[0:3] + "-" + ssnText[3:6] +  "-" + ssnText[6:] 
formatSSNUDF = udf(formatSSN)  

from pyspark.sql.functions import col
exercise_cap_df = exercise_df.select(
                                    firstCapitalizeUDF(col("firstName")).alias("firstName"),
                                    firstCapitalizeUDF(col("middleName")).alias("middleName"),
                                    firstCapitalizeUDF(col("lastName")).alias("lastName"),
                                    firstCapitalizeUDF(col("gender")).alias("gender"),
                                    col("birthdate"),
                                    col("salary"),
                                    formatSSNUDF(col("ssn" )).alias("ssn"))
                                    
                                    

#register the Dataframe so that a SQL can be applied on it

exercise_cap_df.createOrReplaceTempView('exercise_cap_vw')

query = """
select distinct * from exercise_cap_vw
"""
result = spark.sql(query)

result.count()

(3) Spark Jobs
Out[75]: 100000


destFile = userhome + "/people.parquet"

# In case it already exists
result.write.parquet(destFile)
```
