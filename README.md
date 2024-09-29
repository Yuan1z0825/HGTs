# HGTs Repository

Welcome to the **HGTs** repository! This repository is dedicated to the development and exploration of advanced graph-based models, with a focus on Graph Neural Networks (GNNs) and Transformers. The `GraphLab` directory contains the core components of our models, including data loading, model creation, training, and utilities.

## Directory Structure

The `GraphLab` directory is organized into several key components:


- **model/**: This directory houses the core model implementations. It includes various GNNs, Transformer-based models, and other custom architectures.

- **utils/**: Utility scripts that assist in data processing, model training, and evaluation. This includes logging, checkpointing, and other helper functions.

## Key Scripts

- **DeepLoss.py**: Implements the deep loss functions used in training our models (not use).

- **DrawGraph.py**: Contains functions for visualizing graphs. This is useful for understanding the structure of the graph data and the model's learned features.

- **checkpoint.py**: Manages the saving and loading of model checkpoints during training. 

- **cmd_args.py**: Handles command-line arguments for configuring different aspects of the training and model setup. 

- **config.py**: Contains configuration settings, such as model parameters, paths, and other necessary settings.

- **loader.py**: Manages data loading and preprocessing.

- **logger.py**: Provides logging functionality to track training progress, metrics, and other important information.

- **loss.py**: Includes loss functions.

- **model_builder.py**: A utility script for constructing models based on the specified configurations. This helps in creating and initializing the model architecture.

- **optimizer.py**: Defines the optimization strategies used during training, such as SGD, Adam, etc.

- **register.py**: Manages the registration of different model components, ensuring that they can be easily referenced and utilized in the main training script.

- **train.py**: The main training script that integrates all the components and runs the model training process. This script ties together the data loading, model, optimizer, and loss functions.

The `Run` directory in this repository contains essential components and scripts required for executing various graph-based machine learning experiments. Below is a breakdown of the contents:

## Directory Structure

### `CreateGraph/`
This folder contains scripts or modules responsible for the creation and manipulation of graph data structures. It includes functionalities for:
- Generating graphs
- Defining nodes and edges
- Constructing specific types of graphs used in the model or experiments

### `configs/`
This directory stores configuration files (such as YAML or JSON) that define parameters and settings for different runs or experiments. The configuration files allow for:
- Easy modification of experiment parameters
- Replication of experiments with different settings without altering the codebase

### `dataloader.py`
This script is responsible for loading and preprocessing the data used in experiments. 


### `main.py`
The `main.py` file serves as the entry point for running the model or experiments. 
- Set up the experiment
- Initialize the model
- Execute the training and evaluation processes

## Dataset
We utilize the TCGA-LIHC dataset for our experiments. You can download the dataset from the following cloud storage address:
Cloud Storage Address: https://pan.baidu.com/s/12l-pDBlOxUtTK5EVQU4QVw?pwd=ve85 
Extract code：ve85 
## Getting Started

Happy coding!
