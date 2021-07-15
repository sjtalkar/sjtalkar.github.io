# Know for the exam (from the webinars for DP-100)
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
(GPU is reuired by CUDAs)

4. Which datastore cannot be used as a datastore in Azure ML
- Azure DW(Synapse)

5. For visual interface - Create a workspace with 'enterprise' SKU.
Make sure the three basic elements - subscription id, resource group and workspace name











