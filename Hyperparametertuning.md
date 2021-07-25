# Hyperparameter tuning using Azure ML HyperDrive package

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

Azure ML supports **Sampling the hyperparameter space**
Specify the parameter sampling method to use over the hyperparameter space. Azure Machine Learning supports the following methods:

1. Random sampling: In random sampling, hyperparameter values are randomly selected from the defined search space.
2. Grid sampling:  Grid sampling does a simple grid search over all possible values (only choice hyperparameters) . Use grid sampling if you can budget to exhaustively search over the search space.
3. Bayesian sampling: It picks samples based on how previous samples did, so that new samples improve the primary metric.






 
