
### Why DELTA LAKE: A File format that is Open source and Open Storage in Parquet to maintain DATA QUALITY!!!

-Historical queries (Streaming for real time as well as batch jobs and warehouses stored for analytics)
-Messy data (schema adherence challenges)
-Mistakes and Failures (small file sizes/partitions)
-Updates (Upsert requirements)
-Enforce ACID transactions


[Source for intro to Delta Lake](https://www.youtube.com/watch?v=LJtShrQqYZY)
![Why Delta lake](https://github.com/sjtalkar/DP-100AzureSupervisedUnsupervisedDatabricksAndSpark/blob/main/Pictures%20for%20Readme/Deltalakeneed.JPG)

#### Why Full ACID transactions are required?
- No Atomicity means failed Production jobs leave data in corrupt state requiring tedious recovery.
- No quality enforcement create inconsistent and unusable data
- No consistency/isolation makes it almost impossible to mix appends and reads, batch and streaming

#### Levels of data refinement as dataflows through data pipelines in Delta Lake

1. Bronze : Raw ingestion : Dumping ground for raw data often with long retention
2. Silver Filtered, augmented (intermediate data with some cleanup, queryable for easy debugging)
3. Gold (Business level aggregates)

#### Other advantages
>*Two of the core features of Delta Lake are performing upserts (insert/updates) and Time Travel operations. 
>
>* Scalable Metadata Handling: In big data, even the metadata itself can be "big data". Delta Lake treats metadata just like data, leveraging Spark's distributed processing power to handle all its metadata. As a result, Delta Lake can handle petabyte-scale tables with billions of partitions and files at ease.
>
>* Time Travel (data versioning): Delta Lake provides snapshots of data enabling developers to access and revert to earlier versions of data for audits, rollbacks or to reproduce experiments.
>
>* Open Format: All data in Delta Lake is stored in Apache Parquet format enabling Delta Lake to leverage the efficient compression and encoding schemes that are native to Parquet.
>
>* Unified Batch and Streaming Source and Sink: A table in Delta Lake is both a batch table, as well as a streaming source and sink. Streaming data ingest, batch historic backfill, and interactive queries all just work out of the box.
>
>* Schema Enforcement: Delta Lake provides the ability to specify your schema and enforce it. This helps ensure that the data types are correct and required columns are present, preventing bad data from causing data corruption.
>
>* Schema Evolution: Big data is continuously changing. Delta Lake enables you to make changes to a table schema that can be applied automatically, without the need for cumbersome DDL.
>
>* 100% Compatible with Apache Spark API: Developers can use Delta Lake with their existing data pipelines with minimal change as it is fully compatible with Spark, the commonly used big data processing engine


#### To start out with Delta files, say from a dataframe, store it as a Delta file - as you would a Parquet file
```python
# write to Delta Lake in the format delta
rawDataDF.write.mode("overwrite").format("delta").partitionBy("Country").save(DataPath)

# You can read it in, ina similar fashion
new_df = spark.read.format("delta").load(DataPath)

# Some SQL DDL in DELTA

spark.sql("""
  DROP TABLE IF EXISTS customer_data_delta
""")
spark.sql("""
  CREATE TABLE customer_data_delta
  USING DELTA
  LOCATION '{}'
""".format(DataPath))


#Then using cell magic in a .dbc notebook, esplore the data to your heart's content

%sql
SELECT count(*) FROM customer_data_delta

#Note that we only store table name, path, database info in the Hive metastore, the actual schema is stored in the _delta_log directory as shown below.
display(dbutils.fs.ls(DataPath + "/_delta_log"))

#Also get the metadata
%sql
DESCRIBE DETAIL customer_data_delta

# Append newly read data in a dataframe  Note the append mode
(newDataDF
  .write
  .format("delta")
  .partitionBy("Country")
  .mode("append")
  .save(DataPath)
)

#Register it so it can be used in a SQL query
upsertDF.createOrReplaceTempView("upsert_data")

%sql
MERGE INTO customer_data_delta
USING upsert_data
ON customer_data_delta.InvoiceNo = upsert_data.InvoiceNo
  AND customer_data_delta.StockCode = upsert_data.StockCode
WHEN MATCHED THEN
  UPDATE SET *
WHEN NOT MATCHED
  THEN INSERT *

```

#### Batch Operations
```python
upsertDF = spark.read.format("json").load("/mnt/training/enb/commonfiles/upsert-data.json")
display(upsertDF)


>Sources:
> * Microsoft Azure Learning Path
> Databricks comes with default demo datasets  
> [](https://www.youtube.com/watch?v=oXwGFaQOgS0)
> %fs ls “databricks-datasets”
```

**** Recap
> Saving to Delta Lake is as easy as saving to Parquet, but creates an additional log file.
> Using Delta Lake to create tables is straightforward and you do not need to specify schemas.
> With Delta Lake, you can easily append new data without schema-on-read issues.
> Changes to Delta Lake files will immediately be reflected in registered Delta tables.
> Generally, the distinction between tables and DataFrames in Spark can be summarized by discussing scope and persistence:
> Tables are defined at the workspace level and persist between notebooks.
> DataFrames are defined at the notebook level and are ephemeral.


#### FULL SET OF STEPS FROM DATAFRAME TO SPARK SQL TABLE

```python

#Read from a table (if one exists for training) - if not read from CSV, JSON... into a dataframe
deltaDF = spark.read.table('customer_data_delta')


#Write a dataframe into a delta format file out to a location by partitioning it and overwriting if it exists
(customerCounts.write
 .format('delta')
 .partitionBy("Country")
 .mode("overwrite")
 .save(CustomerCountsPath)
 
)
CustomerCountsPath = userhome + "/delta/customer_counts/"

dbutils.fs.rm(CustomerCountsPath, True) #deletes Delta table if previously created

#Register the table as a Spark SQL table
spark.sql("""
  DROP TABLE IF EXISTS customer_counts
""")

spark.sql("""
  CREATE TABLE customer_counts
  USING DELTA
  LOCATION '{}'
""".format(CustomerCountsPath))


#Add additional data from a CSV

newDataPath = "/mnt/training/online_retail/outdoor-products/outdoor-products-small.csv"
newDataDF = (spark
 .read
 .option("header", "true")
 .schema(inputSchema)
  .csv(newDataPath)
            )

```

## SMALL FILE PROBLEM
Historical and new data is often written in very small files and directories.

This data may be spread across a data center or even across the world (that is, not co-located).
The result is that a query on this data may be very slow due to

- network latency
- volume of file metatadata
The solution is to compact many small files into one larger file. Delta Lake has a mechanism for compacting small files.

Delta Lake supports the**OPTIMIZE** operation, which performs file compaction.

**ZORDER** usage
With Delta Lake the notation is:

OPTIMIZE Students
ZORDER BY Gender, Pass_Fail

This will ensure all the data backing Gender = 'M' is colocated, then data associated with Pass_Fail = 'P' is colocated.

See References below for more details on the algorithms behind ZORDER.

Using ZORDER, you can order by multiple columns as a comma separated list; however, the effectiveness of locality drops.

In streaming, where incoming events are inherently ordered (more or less) by event time, use ZORDER to sort by a different column, say 'userID'.
Ensures that all data backing, for example, Grade=8 is colocated, then rewrites the sorted data into new Parquet files

**VACUUM**
To save on storage costs you should occasionally clean up invalid files using the VACUUM command.

Invalid files are small files compacted into a larger file with the OPTIMIZE command.

The syntax of the VACUUM command is

VACUUM name-of-table RETAIN number-of HOURS;

The number-of parameter is the retention interval, specified in hours.


**Time Travel**
Because Delta Lake is version controlled, you have the option to query past versions of the data. Let's look at the history of our current Delta table.

```python
#View the different versions
%sql

DESCRIBE HISTORY customer_data_delta

#Select the version you want
%sql
SELECT COUNT(*)
FROM customer_data_delta
VERSION AS OF 1

#Find the number of new records placed in
%sql
SELECT SUM(total_orders) - (
  SELECT SUM(total_orders)
  FROM customer_counts
  VERSION AS OF 0
  WHERE Country='Sweden') AS new_entries
FROM customer_counts
WHERE Country='Sweden'


