# Know for the exam (from the webinars for DP-100)

## PIPELINE VISUALIZED

![Workspce, dataset, compute, run](https://github.com/sjtalkar/sjtalkar.github.io/blob/main/Run%20A%20Pipeline%20-%201.JPG)


```python
import azureml.core
from azureml.core import Workspace, Experiment, Run
from azureml.core import ScriptRunConfig

ws = Workspace.from_config()
exp = Experiment(workspace = ws, name ="explore-runs")

notebook_run = exp.start_logging()
notbook_run.log(name="message", value="Hello from run!")
```

## Using the CLI

`az ml run submit-script -c sklearn -e testexperiment train.py`

## DataStores and Datasets
![Picture of stores and data drift management](https://github.com/sjtalkar/sjtalkar.github.io/blob/main/DataStoresInAzure.png)

>Tabular Dataset
- Represents data in a tabular fomat by parsing the provided file or list of files
- Lets you materialize the data into a Pandas or Spark Dataframe for further manipulation and cleansing

>File dataset
- References single ot multiple files in your datastores or public URLs
- You can download or mount files of your chooing to your compute target as a FileDataset object

> Both these are versio packaged

## Data Labeling
- Azure ML gives you a central location to create, manage and monitor labeling projects. 
- Labeling projects help coordinate the data and the assigned labels (target labelings) to assist management of labeling the data
- Labeling is required for taks such as image classification - multi-label (can have overlap such as topics within a document) or multi-class (mutially exclusive)


## Data Drift
- It is the change in model input data and predictably can lead to model performance degradation
- Azure datasets can help in detecting the drift

# Experiments

- An **experiment** is a **named process**, usually the running of a script or  pipeline, that can generate metrics and outputs and be tracked in Azure ML workspace.
- An experiment can be run multiple times, with different data, code, or settings.
- in Azure ML, you can track each run and view run history and compare results for each run.

> **Experiment Run Context**
```python
from azureml.core import Experiment
#create an experiment variable
experiment = Experiment(workspace=ws, name="first-experiment")
#start an experiment with the variable
run = experiment.start_logging()

###### CODE THE EXPERIMENT ########
###### ################### ########

#end the experiment
run.complete()
```

### Logging Metrics and Creating Outputs
- __log__ Record a songle **named** value
- __log_list__ Record a named list of values
- __log_row__ Record a row with multiple columns
- __log_table__ Record a dictionary as a table
- __log_image__ Record an image or a plot

```python
data = pd.read_csv("somedata.csv")
row_count = len(data)
run.log("LengthOfDataset", row_count)
#### experiment goes on
```

### Retrieve and viewing above logged metrics
```python
from azureml.widgets import RunDetails
RunDetails(run).show()
```
>Alternatively
```python
import json
metrics = run.get_metrics()
print(json.dumps(metrics, indent=2))
```

### Experiment output **files**
> outputs are logged using upload_file into the outputs folder
```python

os.makedirs('outputs', exists_ok=True)
run.upload_file(name='outputs/sample.csv', path_or_stream='./sample.csv')

```
> When running an experiment in a remote compute context, any files written to the outputs folder in the compute context are automatically uploaded to the run's **outputs** folder when the run completes

> Retrieve these files using run.get_file_names 

## The Run context
```python
##To create
#start an experiment with the variable
run = experiment.start_logging()

## To get existing
from azureml.core import Run
run = Run.get_context()
```

## Experiment as a Script
> You can place the body of the experiment (all the pipelines, evaluation...) into a python script file and use the run context to run the script

```python
from azureml.core import Experiment, RunConfiguration, ScriptRunConfig

#create a new RunConfig object
experiment_run_config = RunConfiguration()

#Create script config
script_config = ScriptRunConfig(source_directory=experiment_folder,
                                script='experiment.py',
                                run_config=experiment_run_config)

#submit the experiment
experiment = Experiment(workspace=ws, name='first_experiment')
run = experiment.submit(config=script_config)
run.wait_for_completion(show_output=True)
```

## Questions from part 1 

1. You want to develop a Machine Learning Model using R. Which application will provide you the tools you need.
Answer - 
- Azure SQL
- SQL Server
- Azure SDK
- 

2. Which configuration options must be set in config.json file
- Subscription
- Resource Group
- workspace_name

3. You are interested in developing your ML modesl using CUDAs. You have also decided you do not need more than 12 cores. Which storage option should you pick?
- Standard_NV12s_v3 with 12VCPUs 1GPU
(GPU is required by CUDAs)

4. Which datastore cannot be used as a datastore in Azure ML
- Azure DW(Synapse)

5. For visual interface - Create a workspace with 'enterprise' SKU.
Make sure the three basic elements - subscription id, resource group and workspace name


# Optimize and manage models

## Contending and dealing with Data Drift

- Set a baseline model for the data by registering a baseline dataset
```python
from azureml.core import Datastore, Dataset


# Upload the baseline data
default_ds = ws.get_default_datastore()
default_ds.upload_files(files=['./data/diabetes.csv', './data/diabetes2.csv'],
                       target_path='diabetes-baseline',
                       overwrite=True, 
                       show_progress=True)

# Create and register the baseline dataset
print('Registering baseline dataset...')
baseline_data_set = Dataset.Tabular.from_delimited_files(path=(default_ds, 'diabetes-baseline/*.csv'))
baseline_data_set = baseline_data_set.register(workspace=ws, 
                           name='diabetes baseline',
                           description='diabetes baseline data',
                           tags = {'format':'CSV'},
                           create_new_version=True)

print('Baseline dataset registered!')
```
>Create a target dataset
Over time, you can collect new data with the same features as your baseline training data. To compare this new data to the baseline data, you must define a target dataset that includes the features you want to analyze for data drift as well as a timestamp field that indicates the point in time when the new data was current -this enables you to measure data drift over temporal intervals. The timestamp can either be a field in the dataset itself, or derived from the folder and filename pattern used to store the data. For example, you might store new data in a folder hierarchy that consists of a folder for the year, containing a folder for the month, which in turn contains a folder for the day; or you might just encode the year, month, and day in the file name like this: data_2020-01-29.csv; 

- Create and register a dataset with a datetime column which has new data
```python
# Upload the files
path_on_datastore = 'diabetes-target'
default_ds.upload_files(files=file_paths,
                       target_path=path_on_datastore,
                       overwrite=True,
                       show_progress=True)

# Use the folder partition format to define a dataset with a 'date' timestamp column
partition_format = path_on_datastore + '/diabetes_{date:yyyy-MM-dd}.csv'
target_data_set = Dataset.Tabular.from_delimited_files(path=(default_ds, path_on_datastore + '/*.csv'),
                                                       partition_format=partition_format)

# Register the target dataset
print('Registering target dataset...')
target_data_set = target_data_set.with_timestamp_columns('date').register(workspace=ws,
                                                                          name='diabetes target',
                                                                          description='diabetes target data',
                                                                          tags = {'format':'CSV'},
                                                                          create_new_version=True)

print('Target dataset registered!')
```



> Create a data drift MONITOR

The data drift monitor will run periodically on-demand to compare the baseline dataset with the target dataset, to ehich new data will be added over time

- Create a compute target to run the data drift monitor
`    training_cluster = ComputeTarget(workspace=ws, name=cluster_name)`
If the above returns no data, then create a compute target
```python
   compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_DS11_V2', max_nodes=2)
        training_cluster = ComputeTarget.create(ws, cluster_name, compute_config)
        training_cluster.wait_for_completion(show_output=True)
```

> Define the data drift monitor : Use the DataDriftDetector class
- Specify the features you want to monitor for data drift
- Name of the compute target to be used to run the monitoring process
- Frequency at which the data should be compared
- The drift threshold abive which an alert should be triggered
- Latency (in hours) to allow for data collection

```python
from azureml.datadrift import DataDriftDetector

# set up feature list
features = ['Pregnancies', 'Age', 'BMI']

# set up data drift detector
monitor = DataDriftDetector.create_from_datasets(ws, 'mslearn-diabates-drift', baseline_data_set, target_data_set,
                                                      compute_target=cluster_name, 
                                                      frequency='Week', 
                                                      feature_list=features, 
                                                      drift_threshold=.3, 
                                                      latency=24)
monitor
```

> Backfill the data drift monitor
Use te baseline model and the target model to analyze data drift between them

``` python
from azureml.widgets import RunDetails

backfill = monitor.backfill(dt.datetime.now() - dt.timedelta(weeks=6), dt.datetime.now())

RunDetails(backfill).show()
backfill.wait_for_completion()
```
>Analyze Drift data
```python
drift_metrics = backfill.get_metrics()
for metric in drift_metrics:
    print(metric, drift_metrics[metric])
```

>You can also visualize the data drift metrics in Azure Machine Learning studio by following these steps:
>On the Datasets page, view the Dataset monitors tab.
> - Click the data drift monitor you want to view.
> - Select the date range over which you want to view data drift metrics (if the column chart does not show multiple weeks of data, wait a minute or so and click Refresh).
> - Examine the charts in the Drift overview section at the top, which show overall drift magnitude and the drift contribution per feature.
> - Explore the charts in the Feature detail section at the bottom, which enable you to see various measures of drift for individual features.


# Detect and Mitigate Unfairness in Models BIAS!!
> Fairlearn package (In preview)
>> [FAIRNESS CHECKLIST](https://www.microsoft.com/en-us/research/publication/co-designing-checklists-to-understand-organizational-challenges-and-opportunities-around-fairness-in-ai/)


> You begin by setting a range or creating group ranges so that metrics can be compared for the groups/ranges
```python
# Change value to represent age groups
S['Age'] = np.where(S.Age > 50, 'Over 50', '50 or younger')
```

> You can then create a MetricsFrame for accurracy, precision, recall (if classification) as seen below

```python
!pip show azureml-contrib-fairness
!pip install --upgrade fairlearn==0.7.0 raiwidgets


### NOTE THE BELOW
from fairlearn.metrics import selection_rate, MetricFrame


from sklearn.metrics import accuracy_score, recall_score, precision_score


from fairlearn.metrics import selection_rate, MetricFrame
from sklearn.metrics import accuracy_score, recall_score, precision_score

# Get predictions for the witheld test data
y_hat = diabetes_model.predict(X_test)

# Get overall metrics
print("Overall Metrics:")
# Get selection rate from fairlearn
overall_selection_rate = selection_rate(y_test, y_hat) # Get selection rate from fairlearn
print("\tSelection Rate:", overall_selection_rate)
# Get standard metrics from scikit-learn
overall_accuracy = accuracy_score(y_test, y_hat)
print("\tAccuracy:", overall_accuracy)
overall_recall = recall_score(y_test, y_hat)
print("\tRecall:", overall_recall)
overall_precision = precision_score(y_test, y_hat)
print("\tPrecision:", overall_precision)


### NOTE THAT THE METRICS ARE BEING DEFINED FOR EACH OF THE GROUPS YOU ARE INTERESTED IN, SEPARATELY!!!
# Get metrics by sensitive group from fairlearn
print('\nMetrics by Group:')
metrics = {'selection_rate': selection_rate,
           'accuracy': accuracy_score,
           'recall': recall_score,
           'precision': precision_score}

group_metrics = MetricFrame(metrics=metrics,
                             y_true=y_test,
                             y_pred=y_hat,
                             sensitive_features=S_test['Age'])

print(group_metrics.by_group)


```
> Metrics by Group:
              selection_rate  accuracy    recall precision
Age                                                       
50 or younger       0.301491  0.893981  0.827778  0.818681
Over 50             0.714286   0.89418  0.945736  0.903704

> Selection rate means the fraction of datapoints in each class classified as 1 (in binary classification) or distribution of prediction values (in regression).



# Registering the built model
```python
from azureml.core import Workspace, Experiment, Model
import joblib
import os

# Load the Azure ML workspace from the saved config file
ws = Workspace.from_config()
print('Ready to work with', ws.name)

# Save the trained model
model_file = 'diabetes_model.pkl'
joblib.dump(value=diabetes_model, filename=model_file)

# Register the model
print('Registering model...')
registered_model = Model.register(model_path=model_file,
                                  model_name='diabetes_classifier',
                                  workspace=ws)
model_id= registered_model.id


print('Model registered.', model_id)
```










