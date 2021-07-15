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

az ml run submit-script -c sklearn -e testexperiment train.py




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

