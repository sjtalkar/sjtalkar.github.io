# DP-100
## Machine Learning with Spark

[DP-100 ML with Spark](https://docs.microsoft.com/en-us/learn/modules/perform-machine-learning-with-azure-databricks/2-understand)

>NOTE: Sources for below Read Me text and pictures
       - Azure Databricks provides a large number of datasets. Access them in a .dbc with:

```%fs ls “databricks-datasets”    ```

Spark differs from many other machine learning frameworks in that we train our model on a single column that contains a vector of all of our features. Prepare the data by creating one column named features that has the average number of rooms, crime rate, and poverty percentage, for instance.

VectorAssembler : A feature transformer that merges multiple columns into a vector column.
[Spark VectorAssembler](https://spark.apache.org/docs/3.1.1/api/python/reference/api/pyspark.ml.feature.VectorAssembler.html)

> Metrics
This section describes the metrics returned for the specific types of models supported for use with Evaluate Model:

classification models
regression models
clustering models
Metrics for classification models
The following metrics are reported when evaluating binary classification models.

Accuracy measures the goodness of a classification model as the proportion of true results to total cases.

Precision is the proportion of true results over all positive results. Precision = TP/(TP+FP)

Recall is the fraction of the total amount of relevant instances that were actually retrieved. Recall = TP/(TP+FN)

F1 score is computed as the weighted average of precision and recall between 0 and 1, where the ideal F1 score value is 1.

AUC measures the area under the curve plotted with true positives on the y axis and false positives on the x axis. This metric is useful because it provides a single number that lets you compare models of different types. AUC is classification-threshold-invariant. It measures the quality of the model's predictions irrespective of what classification threshold is chosen.

Metrics for regression models
The metrics returned for regression models are designed to estimate the amount of error. A model is considered to fit the data well if the difference between observed and predicted values is small. However, looking at the pattern of the residuals (the difference between any one predicted point and its corresponding actual value) can tell you a lot about potential bias in the model.

The following metrics are reported for evaluating regression models.

Mean absolute error (MAE) measures how close the predictions are to the actual outcomes; thus, a lower score is better.

Root mean squared error (RMSE) creates a single value that summarizes the error in the model. By squaring the difference, the metric disregards the difference between over-prediction and under-prediction.

Relative absolute error (RAE) is the relative absolute difference between expected and actual values; relative because the mean difference is divided by the arithmetic mean.

Relative squared error (RSE) similarly normalizes the total squared error of the predicted values by dividing by the total squared error of the actual values.

Coefficient of determination, often referred to as R2, represents the predictive power of the model as a value between 0 and 1. Zero means the model is random (explains nothing); 1 means there is a perfect fit. However, caution should be used in interpreting R2 values, as low values can be entirely normal and high values can be suspect.

Metrics for clustering models
Because clustering models differ significantly from classification and regression models in many respects, Evaluate Model also returns a different set of statistics for clustering models.

The statistics returned for a clustering model describe how many data points were assigned to each cluster, the amount of separation between clusters, and how tightly the data points are bunched within each cluster.

The statistics for the clustering model are averaged over the entire dataset, with additional rows containing the statistics per cluster.

The following metrics are reported for evaluating clustering models.

The scores in the column, Average Distance to Other Center, represent how close, on average, each point in the cluster is to the centroids of all other clusters.

The scores in the column, Average Distance to Cluster Center, represent the closeness of all points in a cluster to the centroid of that cluster.

The Number of Points column shows how many data points were assigned to each cluster, along with the total overall number of data points in any cluster.

If the number of data points assigned to clusters is less than the total number of data points available, it means that the data points could not be assigned to a cluster.

The scores in the column, Maximal Distance to Cluster Center, represent the max of the distances between each point and the centroid of that point's cluster.

If this number is high, it can mean that the cluster is widely dispersed. You should review this statistic together with the Average Distance to Cluster Center to determine the cluster's spread.

The Combined Evaluation score at the bottom of the each section of results lists the averaged scores for the clusters created in that particular model.
>




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
```

## ML Workflows

```
bostonDF = (spark.read
  .option("HEADER", True)
  .option("inferSchema", True)
  .csv("/mnt/training/bostonhousing/bostonhousing/bostonhousing.csv")
)

display(bostonDF)

# Split into Train and test
trainDF, testDF = bostonDF.randomSplit([0.8, 0.2], seed=42)

#Find the average of  the target variable
from pyspark.sql.functions import avg

trainAvg = trainDF.select(avg("medv")).first()[0]
print("Average home value: {}".format(trainAvg))

# The average is our baseline model
from pyspark.sql.functions import lit
testPredictionDF = testDF.withColumn("prediction", lit(trainAvg))

display(testPredictionDF)

#Set a evaluator to score our model The scoring method is MSE

from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator(predictionCol="prediction", 
labelCol="medv", metricName="mse")










