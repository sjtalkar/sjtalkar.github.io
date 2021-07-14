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
