## Workspaces for Machine Learning Assets
>A workspace defines the boundary for a set of related machine learning assets. You can use workspaces to group machine learning assets based on projects, deployment environments (for example, test and production), teams, or some other organizing principle. The assets in a workspace include:

- Compute targets for development, training, and deployment.
- Data for experimentation and model training.
- Notebooks containing shared code and documentation.
- Experiments, including run history with logged metrics and outputs.
- Pipelines that define orchestrated multi-step processes.
- Models that you have trained.

Workspaces as Azure Resources
Workspaces are Azure resources, and as such they are defined within a resource group in an Azure subscription, along with other related Azure resources that are required to support the workspace.

### Not within the workspace but supporting it :
The Azure resources created alongside a workspace include:

- A storage account - used to store files used by the workspace as well as data for experiments and model training.
- An Application Insights instance, used to monitor predictive services in the workspace.
- An Azure Key Vault instance, used to manage secrets such as authentication keys and credentials used by the workspace.
- A container registry, created as-needed to manage containers for deployed models.

### Role-Based Access Control
>You can assign role-based authorization policies to a workspace, enabling you to manage permissions that restrict what actions specific Azure Active Directory (AAD) principals can perform. For example, you could create a policy that allows only users in the IT Operations group to create compute targets and datastores, while allowing users in the Data Scientists group to create and run experiments and register models.

### Creating a Workspace
You can create a workspace in any of the following ways:

- In the Microsoft Azure portal, create a new Machine Learning resource, specifying the subscription, resource group and workspace name.
- Use the Azure Machine Learning Python SDK to run code that creates a workspace.
- Use the Azure Command Line Interface (CLI) with the Azure Machine Learning CLI extension.
- Create an Azure Resource Manager template. For more information the template format for an Azure Machine Learning workspace, see the Azure Machine Learning documentation.

### Azure Machine Learning studio is a web-based tool for managing an Azure Machine Learning workspace. 
It enables you to create, manage, and view all of the assets in your workspace and provides the following graphical tools:

- Designer: A drag and drop interface for "no code" machine learning model development.
- Automated Machine Learning: A wizard interface that enables you to train a model using a combination of algorithms and data preprocessing techniques to find the best model for your data.

## The Azure Machine Learning SDK
While graphical interfaces like Azure Machine Learning studio make it easy to create and manage machine learning assets, it is often advantageous to use a code-based approach to managing resources. By writing scripts to create and manage resources, you can:

- Run machine learning operations from your preferred development environment.
- Automate asset creation and configuration to make it repeatable.
- Ensure consistency for resources that must be replicated in multiple environments (for example, development, test, and production)
- Incorporate machine learning asset configuration into developer operations (DevOps) workflows, such as continuous integration / continuous deployment (CI/CD) pipelines.

### Installing the Azure ML SDK and interactive Jupyter notebook widgets

`pip install azureml-sdk azureml-widgets`

#### Config file
```json
{
    "subscription_id": "1234567-abcde-890-fgh...",
    "resource_group": "aml-resources",
    "workspace_name": "aml-workspace"
}
```


#### Options to connect to the Workspace:

```python
from azureml.core import Workspace
ws = Workspace.from_config()

or
from azureml.core import Workspace

ws = Workspace.get(name='aml-workspace',
                   subscription_id='1234567-abcde-890-fgh...',
                   resource_group='aml-resources')
```
>The SDK contains a rich library of classes that you can use to create, manage, and use many kinds of asset in an Azure Machine Learning workspace.

### Compute Instances
- Compute Instances include Jupyter Notebook and JupyterLab installations that you can use to write and run code that uses the Azure Machine Learning SDK to work with assets in your workspace.
- You can choose a compute instance image that provides the compute specification you need, from small CPU-only VMs to large GPU-enabled workstations.
- You can store notebooks independently in workspace storage, and open them in any compute instance.
- Because compute instances are hosted in Azure, you only pay for the compute resources when they are running; so you can create a compute instance to suit your needs, and stop it when your workload has completed to minimize costs.


## Experiments within a workspace
```python
from azureml.core import Experiment

# create an experiment variable
experiment = Experiment(workspace = ws, name = "my-experiment")

# start the experiment
run = experiment.start_logging()

# experiment code goes here

# end the experiment
run.complete()
```

### You can Log Metric using run command
- log: Record a single named value.
- log_list: Record a named list of values.
- log_row: Record a row with multiple columns.
- log_table: Record a dictionary as a table.
- log_image: Record an image file or a plot.

### Retrieve logged metrics
```python
from azureml.widgets import RunDetails

RunDetails(run).show()
```

>An experiment script is just a Python code file that contains the code you want to run in the experiment. To access the experiment run context (which is needed to log metrics) the script must import the azureml.core.Run class and call its get_context method. The script can then use the run context to log metrics, upload files, and complete the experiment, as shown in the following example:
```python
from azureml.core import Run
import pandas as pd
import matplotlib.pyplot as plt
import os

# Get the experiment run context
run = Run.get_context()

# load the diabetes dataset
data = pd.read_csv('data.csv')

# Count the rows and log the result
row_count = (len(data))
run.log('observations', row_count)

# Save a sample of the data
os.makedirs('outputs', exist_ok=True)
data.sample(100).to_csv("outputs/sample.csv", index=False, header=True)

# Complete the run
run.complete()
```
>
>An implicitly created RunConfiguration object defines the Python environment for the experiment, including the packages available to the script. If your script depends on packages that are not included in the default environment, you must associate the ScriptRunConfig with an Environment object that makes use of a CondaDependencies object to specify the Python packages required. Runtime environments are discussed in more detail later in this course.

### RUN object and various ways to log
> Create an Azure ML experiment in your workspace
experiment = Experiment(workspace=ws, name="mslearn-diabetes")
> Start logging data from the experiment, obtaining a reference to the experiment run
```python
run = Run.get_context()
or
run = experiment.start_logging()
```
>Examples of items that can be logged
```python
run.log('observations', row_count)

run.log_image(name='label distribution', plot=fig)

pregnancies = data.Pregnancies.unique()
run.log_list('pregnancy categories', pregnancies)

```

### How to view the logged items
>In Jupyter Notebooks, you can use the RunDetails widget to see a visualization of the run details.
```python
from azureml.widgets import RunDetails
RunDetails(run).show()
```

When you run the above you will find all that you charted appear. To the top right you can click on View Run Details and a new window will open with these tabs:

-The Details tab contains the general properties of the experiment run.
-The Metrics tab enables you to select logged metrics and view them as tables or charts.
-The Images tab enables you to select and view any images or plots that were logged in the experiment (in this case, the Label Distribution plot)
-The Child Runs tab lists any child runs (in this experiment there are none).
-The Outputs + Logs tab shows the output or log files generated by the experiment.
-The Snapshot tab contains all files in the folder where the experiment code was run (in this case, everything in the same folder as this notebook).
-The Explanations tab is used to show model explanations generated by the experiment (in this case, there are none).
-The Fairness tab is used to visualize predictive performance disparities that help you evaluate the fairness of machine learning models (in this case, there are none).

>You can download the files produced by the experiment, either individually by using the download_file method, or by using the download_files method to retrieve multiple files. The following code downloads all of the files in the run's output folder:
```python
import os

download_folder = 'downloaded-files'

# Download files in the "outputs" folder
run.download_files(prefix='outputs', output_directory=download_folder)

# Verify the files have been downloaded
for root, directories, filenames in os.walk(download_folder): 
    for filename in filenames:  
        print (os.path.join(root,filename))
        ```

### Store a script as a .py file and run
>Running the script as an experiment
To run the script, create a ScriptRunConfig that references the folder and script file. You generally also need to define a Python (Conda) environment that includes any packages required by the script. In this example, the script uses Scikit-Learn so you must create an environment that includes that. The script also uses Azure Machine Learning to log metrics, so you need to remember to include the azureml-defaults package in the environment.

```python
from azureml.core import Experiment, ScriptRunConfig, Environment
from azureml.core.conda_dependencies import CondaDependencies

# Create a Python environment for the experiment
sklearn_env = Environment("sklearn-env")

# Ensure the required packages are installed
packages = CondaDependencies.create(conda_packages=['scikit-learn','pip'],
                                    pip_packages=['azureml-defaults'])
sklearn_env.python.conda_dependencies = packages

# Create a script config
script_config = ScriptRunConfig(source_directory='training_folder',
                                script='training.py',
                                environment=sklearn_env) 

# Submit the experiment
experiment = Experiment(workspace=ws, name='training-experiment')
run = experiment.submit(config=script_config)
run.wait_for_completion()
```



### Types of compute instances
- Compute instances: Development workstations that data scientists can use to work with data and models.
- Compute clusters: Scalable clusters of virtual machines for on-demand processing of experiment code.
- Inference clusters: Deployment targets for predictive services that use your trained models.
- Attached compute: Links to other Azure compute resources, such as Virtual Machines or Azure Databricks clusters.