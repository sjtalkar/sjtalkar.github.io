# DP-100
## Machine Learning with Spark

[DP-100 ML with Spark](https://docs.microsoft.com/en-us/learn/modules/perform-machine-learning-with-azure-databricks/2-understand)

>NOTE: Sources for below Read Me text and pictures
       - Azure Databricks provides a large number of datasets. Access them in a .dbc with:
       ```python
        %fs ls “databricks-datasets”
         ```

Spark differs from many other machine learning frameworks in that we train our model on a single column that contains a vector of all of our features. Prepare the data by creating one column named features that has the average number of rooms, crime rate, and poverty percentage, for instance.

VectorAssembler : A feature transformer that merges multiple columns into a vector column.
[Spark VectorAssembler](https://spark.apache.org/docs/3.1.1/api/python/reference/api/pyspark.ml.feature.VectorAssembler.html)

```python
from pyspark.ml.feature import VectorAssembler

featureCols = ["rm", "crim", "lstat"]
assembler = VectorAssembler(inputCols=featureCols, outputCol="features")

bostonFeaturizedDF = assembler.transform(bostonDF)


from pyspark.ml.regression import LinearRegression

lr = LinearRegression(labelCol="medv", featuresCol="features")

lrModel = lr.fit(bostonFeaturizedDF)

# get the coefficents  for each feature and the intercept of the linear model created
print("Coefficients: {0:.1f}, {1:.1f}, {2:.1f}".format(*lrModel.coefficients))
print("Intercept: {0:.1f}".format(lrModel.intercept))


# To predict for a new datapoint (vector of features)

from pyspark.ml.linalg import Vectors

data = [(Vectors.dense([6., 3.6, 12.]), )]              # Creates our hypothetical data point
predictDF = spark.createDataFrame(data, ["features"])

display(lrModel.transform(predictDF))


'''





