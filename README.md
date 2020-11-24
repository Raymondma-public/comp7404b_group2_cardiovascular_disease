# comp7404b_group2_life_expectancy
COMP7404B Group project(Group 2). Regional Life Expectancy Regression with Machine Learning

There are factors to the average life expectancy of a population, such as wealth, immunization, and education. We aim to discover the most significant factors and make predictions on the average life expectancy. Various machine learning techniques, like random forest and neural network regression, will be used to build the regression model and make predictions. Non-machine learning models would be used as a benchmark. Statistics Training datasets are openly available on WHO website and kaggle.com.

# Usage
Requirements:
-   Python: 3.7 or above
-   Other [requirements](requirements.txt)

## Set up
1.  Set up virtual environment. For example:
```bash
virtualenv .venv --python=python3
source .venv/bin/activate
```

2.  Install [requirements](requirements.txt).
```bash
pip install -r requirements.txt
```

## Try our demo

The following demos are examples on how to build the training and testing sets. Example of how to use the functions in [utils.py](utils.py) are also shown.

-   [Random Forest Regression](random_forest.py)
-   [Neural Network Regression](multilayer_perception.py)

```bash
python random_forest.py
python multilayer_perception.py
```

## [utils.py](utils.py)

This file includes helper functions for the regression tasks. The functionality of the functions are described in the file.