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

## Getting Started

To get started with this repository, clone the repository and follow the instructions provided in the `README.md` file. You'll find detailed guides on setting up your environment, preparing your data, and running experiments with the models provided.

We hope this repository helps you in your research and development of graph-based models. If you encounter any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.

Happy coding!
