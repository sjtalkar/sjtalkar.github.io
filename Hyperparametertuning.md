# [Hyperparameter tuning using Azure ML HyperDrive package](https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/ml-frameworks/pytorch/train-hyperparameter-tune-deploy-with-pytorch/train-hyperparameter-tune-deploy-with-pytorch.ipynb)

1. Define the parameter search space (Dictionary of parameters) 
2. Specify a **primary metric** to optimize
3. Specify **early termination policy**  for low-performing runs
4. Create and assign resources
5. Launch an experiment with the defined configuration
6. Visualize the training runs
7. Select the best configuration for your model

Model parameters are parameters learnt when training a model from given data. Hyperparameters are adjustable parameters that let you control the model training process 
- for instance number of hidden layers and number of nodes in each layer of a neural network.

Hyperparameter tuning, also called hyperparameter optimization, is the process of finding the configuration of hyperparameters that results in the best performance.
The process is typically computationally expensive and manual.

## DEFINING THE PARAMETERS
- Discrete using **choice** 
```python
{
        "batch_size": choice(16, 32, 64, 128)
        "number_of_hidden_layers": choice(range(1,5))
 }
```
- Continuous  using **uniform**, **normal**,  **loguniform**, **lognormal**
```python
 {    
        "learning_rate": normal(10, 3),
        "keep_probability": uniform(0.05, 0.1)
 }
```

## Azure ML supports **Sampling the hyperparameter space**
Specify the parameter sampling method to use over the hyperparameter space. Azure Machine Learning supports the following methods:

1. **Random** sampling: In random sampling, hyperparameter values are randomly selected from the defined search space.
2. **Grid** sampling:  Grid sampling does a simple grid search over all possible values (only choice hyperparameters) . Use grid sampling if you can budget to exhaustively search over the search space.
3. **Bayesian** sampling: It picks samples based on how previous samples did, so that new samples improve the primary metric.

## Specifying the primary metric

**primary_metric_name**: The name of the primary metric needs to exactly match the name of the metric logged by the training script
**primary_metric_goal**: It can be either PrimaryMetricGoal.**MAXIMIZE** or PrimaryMetricGoal.**MINIMIZE** and determines whether the primary metric will be maximized or minimized when evaluating the runs.
```python
primary_metric_name="accuracy",
primary_metric_goal=PrimaryMetricGoal.MAXIMIZE
```

## Log metric for hyperparameter tuning
The training script for your model must log the primary metric during model training so that HyperDrive can access it for hyperparameter tuning.

```python
from azureml.core.run import Run
run_logger = Run.get_context()
run_logger.log("accuracy", float(val_accuracy))
```

## Specify early termination policy 

- Bandit policy:based on slack factor/slack amount and evaluation interval. Bandit ends runs when the primary metric isn't within the specified slack factor/slack amount of the most successful run.
- Median stopping policy: based on running averages of primary metrics reported by the runs. This policy computes running averages across all training runs and stops runs whose primary metric value is worse than the median of the averages.
- Truncation selection policy :  cancels a percentage of lowest performing runs at each evaluation interval. Runs are compared using the primary metric.
- No termination policy : the hyperparameter tuning service will let all training runs execute to completion.
`Bayesian sampling does not support early termination. When using Bayesian sampling, set early_termination_policy = None.`

## Hints
For a conservative policy that provides savings without terminating promising jobs, consider a Median Stopping Policy with evaluation_interval 1 and delay_evaluation 5. These are conservative settings, that can provide approximately 25%-35% savings with no loss on primary metric (based on our evaluation data).

For more aggressive savings, use Bandit Policy with a smaller allowable slack or Truncation Selection Policy with a larger truncation percentage.

## Configure hyperparameter tuning experiment
To configure your hyperparameter tuning experiment, provide the following:

- The defined hyperparameter search space
- Your early termination policy
- The primary metric
- Resource allocation settings
- **ScriptRunConfig**  script_run_config

```python
from azureml.train.hyperdrive import HyperDriveConfig
from azureml.train.hyperdrive import RandomParameterSampling, BanditPolicy, uniform, PrimaryMetricGoal

param_sampling = RandomParameterSampling( {
        'learning_rate': uniform(0.0005, 0.005),
        'momentum': uniform(0.9, 0.99)
    }
)

early_termination_policy = BanditPolicy(slack_factor=0.15, evaluation_interval=1, delay_evaluation=10)

hd_config = HyperDriveConfig(run_config=script_run_config,
                             hyperparameter_sampling=param_sampling,
                             policy=early_termination_policy,
                             primary_metric_name="accuracy",
                             primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                             max_total_runs=100,
                             max_concurrent_runs=4)
```
The HyperDriveConfig sets the parameters passed to the **ScriptRunConfig** script_run_config. The script_run_config, in turn, passes parameters to the **training script**.





 
