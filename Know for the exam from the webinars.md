# Know for the exam (from the webinars for DP-100)

import azureml.core
from azureml.core import Workspace, Experiment, Run
from azureml.core import ScriptRunConfig

ws = Workspace.from_config()
exp = Experiment(workspace = ws, name ="explore-runs")

notebook_run = exp.start_logging()
notbook_run.log(name="message", value="Hello from run!")


## Using the CLI

az ml run submit-script -c sklearn -e testexperiment train.py


