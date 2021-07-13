## MLflow Tracking

>MLflow Tracking
MLflow Tracking is a logging API specific for machine learning and agnostic to libraries and environments that do the training. It is organized around the concept of runs, which are executions of data science code. Runs are aggregated into experiments where many runs can be a part of a given experiment and an MLflow server can host many experiments.

[Managing the complete ML lifecycle with MLflow](https://www.youtube.com/watch?v=x3cxvsUFVZA)

MLflow tracking also serves as a model registry so tracked models can easily be stored and, as necessary, deployed into production. This also standardizes this process, which significantly accelerates it and allows for scalability. Experiments can be tracked using libraries in Python, R, and Java as well as by using the CLI and REST calls. This module will use Python, though the majority of MLflow functionality is also exposed in these other APIs.

### Track Runs
Each run can record the following information:

- Parameters: Key-value pairs of input parameters such as the number of trees in a random forest model
- Metrics: Evaluation metrics such as RMSE or Area Under the ROC Curve
- Artifacts: Arbitrary output files in any format. This can include images, pickled models, and data files
- Source: The code that originally ran the experiment

1. To tune parameters
2. For data governance
3. Reproduce the results
4. Interpretability
5. Scalable


`NOTE: MLflow can only log PipelineModels. Saved models get an id.`

```python
%python

loaded_model = mlflow.spark.load_model(f"runs:/{run.info.run_uuid}/log-model")
display(loaded_model.transform(testDF))
```
![ML Lifecycle and tools](https://github.com/sjtalkar/sjtalkar.github.io/blob/main/MLLearningLifecycle.JPG)