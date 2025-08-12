# Evaluation Directory Overview

This directory is designed to evaluate various models and methods, with results stored in the `res` folder.

## eval_on_bdd.py
This script is used to evaluate a model on a predefined dataset.  
- **Setup**: Fill the `MODEL` variable with the name of the model folder (e.g., `20250630_225812_yhajndhh`).  
- **Configuration**: The `cfg` variable contains configuration parameters. Avoid modifying these for better comparison across models.

## eval.py
This script evaluates models either quickly or on generated datasets with varying client sizes.  
- **Setup**:  
  - `FOLDER`: Specify the folder containing the models to test (e.g., all models from the same experiment).  
  - `rapid`: Set to `True` for quick evaluation or `False` for full evaluation.  
  - `dim`: Represents the size of instances for testing. Ignored if `rapid` is `True`.

## func.py
Contains utility functions used across different scripts. Examples include initializing problem parameters, setting random seeds, loading models, and plotting solutions.

## generate_res_plot.py
Generates a comparative graph of the tested models.  
- **Purpose**: Visualize and compare the performance of different models based on their results.

## visu_exp.py
Visualizes data from a specific experiment.  
- **Setup**: Fill the `MODEL_NAME` variable with the name of the model to use for visualization.
