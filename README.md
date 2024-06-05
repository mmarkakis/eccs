# ECCS: Exposing Critical Causal Structures

Welcome to the repository for the ECCS project! You can access the documentation [here](https://mmarkakis.github.io/eccs/).

For technical details about the project, you can [read our paper](https://people.csail.mit.edu/markakis/papers/2024_ECCS.pdf). 

## Table of Contents

1. [Setting up a virtual environment and installing dependencies](#1-setting-up-a-virtual-environment-and-installing-dependencies).

2. [Reproducing our evaluation](#2-reproducing-our-evaluation)

3. [Rebuilding the documentation](#3-rebuilding-the-documentation)


## 1. Setting up a virtual environment and installing dependencies

Using a virtual environment is recommended to ensure dependencies are managed correctly. This section will walk you through setting up a virtual environment for this project. Before starting, make sure you have:

- Python 3 installed on your system
- Access to the command line/terminal


### 1.1. Creating the Virtual Environment

First, navigate to the project's root directory in your terminal. Then, create a virtual environment by running:

```bash
python3 -m venv eccs-venv
```

This command creates a new directory `eccs-venv` in your project where the virtual environment files are stored.

### 1.2. Activating the Virtual Environment

To activate the virtual environment, use the following command:

On Windows:
```cmd
.\eccs-venv\Scripts\activate
```

On macOS and Linux:
```bash
source eccs-venv/bin/activate
```

After activation, your terminal prompt will change to indicate that the virtual environment is active.

### 1.3. Installing Dependencies

With the virtual environment active, install the project dependencies by running:

``` bash
pip install -r requirements.txt
```

### 1.4. Deactivating the Virtual Environment
When you're done working in the virtual environment, you can deactivate it by running:

```bash
deactivate
```

This command will return you to your system's default Python interpreter.

## 2. Reproducing our evaluation

Reproducing our evaluation is super easy! Just run the following command from the root of this repository (within the virtual environment you created above):

```bash
python3 src/evaluation/iterative_runner.py
```

An experimental directory will be created under `evaluation/`, named after the current timestamp `<ts>`. After the experimental run completes, you will be able to find plots like the ones included in [Figure 2 of our paper](https://people.csail.mit.edu/markakis/papers/2024_ECCS.pdf) under `evaluation/<ts>/plots/`. Note that each experimental run creates new ground truth causal graphs, datasets, and starting causal graphs, so your plots may vary from the results in the paper.

You can edit `src/evaluation/iterative_config.yml` to adjust any experimental parameters.

NOTE: Running all of the experiments in our evaluation can take several hours, depending on your hardware. You may want to use a tool like [tmux](https://github.com/tmux/tmux/wiki) to run the above command in the background.


## 3. Rebuilding the documentation

To rebuild the documentation after editing the code, you can run:

```bash
mkdocs gh-deploy
```


