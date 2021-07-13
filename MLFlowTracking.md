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


## Tracking server

![Protocol and Cloud Server](https://github.com/sjtalkar/sjtalkar.github.io/blob/main/TrackingServer.JPG)

### Advantages and utilities of MLflow
- Modular Components greatly simplify the ML lifecycle
- Easy to install and use 
- Develop and deploy locally, but track locally or remotely
- Available APIs: Python, Java and R 
- Visualize experiments and compare runs
- Centrally register and manage model lifecycle

[Tutorials](https://dbricks.co/mlflow-part-1)

### Exercise
[Gas Consumption Prediction](https://github.com/dmatrix/mlflow-workshop-part-1/blob/master/notebooks/MLflow-CE.dbc) 


### Logging in MLflow
- mlflow.sklearn.log_model
- mlflow.log_params()
- mlflow.log_metric()


### This is the core of how a model was saved 

You can find this code in : rfr_regression_base_exp_cls (inside setup folder in the tutorials folder called MLflow-CE)

```python
 def mlflow_run(self, df, r_name="Lab-4:RF Experiment Model"):
        """
        Override the base class mlflow_run for this epxerimental runs
        This method trains the model, evaluates, computes the metrics, logs
        all the relevant metrics, artifacts, and models.
        :param df: pandas dataFrame
        :param r_name: name of the experiment run
        :return:  MLflow Tuple (ExperimentID, runID)
        """

        with mlflow.start_run(run_name=r_name) as run:
            # get experimentalID and runID
            runID = run.info.run_uuid
            experimentID = run.info.experiment_id
            
            # split train/test and train the model
            X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1), df[["price"]].values.ravel(), random_state=42)
            self._rf.fit(X_train, y_train)
            predictions = self._rf.predict(X_test)
            # create an Actual vs Predicted DataFrame
            df_preds = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': predictions.flatten()})

            # Log model and parameters
            mlflow.sklearn.log_model(self.model, "random-forest-model")

            # Note we are logging as a dictionary of all params instead of logging each parameter
            mlflow.log_params(self.params)


            # Create metrics
            mse = metrics.mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            mae = metrics.mean_absolute_error(y_test, predictions)
            r2 = metrics.r2_score(y_test, predictions)

            # Log metrics
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            
            # Log prediciton/actual values in file as a feature artifact
            temp_file_name = Utils.get_temporary_directory_path("predicted-actual-", ".csv")
            temp_name = temp_file_name.name
            try:
                df_preds.to_csv(temp_name, index=False)
                mlflow.log_artifact(temp_name, "predicted-actual-files")
            finally:
                temp_file_name.close()  # Delete the temp file

            # Create feature importance and save them as artifact
            # This allows us to remove least important features from the dataset
            # with each iteration if they don't have any effect on the predictive power of
            # the prediction.
            importance = pd.DataFrame(list(zip(df.columns, self._rf.feature_importances_)),
                                      columns=["Feature", "Importance"]
                                      ).sort_values("Importance", ascending=False)

            # Log importance file as feature artifact
            temp_file_name = Utils.get_temporary_directory_path("feature-importance-", ".csv")
            temp_name = temp_file_name.name
            try:
                importance.to_csv(temp_name, index=False)
                mlflow.log_artifact(temp_name, "feature-importance-files")
            finally:
                temp_file_name.close()  # Delete the temp file

            # Create residual plots and image directory
            # Residuals R = observed value - predicted value
            (plt, fig, ax) = Utils.plot_residual_graphs(predictions, y_test, "Predicted values for Price ($)", "Residual",
                                                  "Residual Plot")

            # Log residuals images
            temp_file_name = Utils.get_temporary_directory_path("residuals-", ".png")
            temp_name = temp_file_name.name
            try:
                fig.savefig(temp_name)
                mlflow.log_artifact(temp_name, "residuals-plots")
            finally:
                temp_file_name.close()  # Delete the temp file

            print("-" * 100)
            print("Inside MLflow {} Run with run_id {} and experiment_id {}".format(r_name, runID, experimentID))
            print("  mse: {}".format(mse))
            print(" rmse: {}".format(rmse))
            print("  mae: {}".format(mae))
            print("  R2 : {}".format(r2))

            return (experimentID, runID)
   
 ```  

### Analyzing run in the Experiment UI.


![At the top right corner, the meny item Experiments when clicked will offer the details about each run](https://github.com/sjtalkar/sjtalkar.github.io/blob/main/AnalysingRuns.JPG)


The class_setup module retrieves the stored model
```python
# iterate over several runs with different parameters, such as number of trees. 
# For expermientation, try max_depth and consult the documentation what tunning parameters
# may affect a better outcome.
max_depth = 0
for n in range (20, 250, 50):
  max_depth = max_depth + 2
  params = {"n_estimators": n, "max_depth": max_depth}
  rfr = RFRModel.new_instance(params)
  (experimentID, runID) = rfr.mlflow_run(dataset)
  print("MLflow Run completed with run_id {} and experiment_id {}".format(runID, experimentID))
  print("-" * 100)
  ```


  #### Other resources
  Some Resources:

- https://mlflow.org/docs/latest/python_api/mlflow.html
- https://www.saedsayad.com/decision_tree_reg.htm
- https://towardsdatascience.com/understanding-random-forest-58381e0602d2
- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
- https://towardsdatascience.com/regression-an-explanation-of-regression-metrics-and-what-can-go-wrong-a39a9793d914
- https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/