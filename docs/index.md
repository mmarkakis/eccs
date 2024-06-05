# ECCS: Exposing Critical Causal Structures

Welcome to the repository for the ECCS project! For technical details about the project, [read our paper here](https://people.csail.mit.edu/markakis/papers/2024_ECCS.pdf).

## Table of Contents

1. [Setting up a virtual environment and installing dependencies](#1-setting-up-a-virtual-environment-and-installing-dependencies).

2. [Reading the documentation](#2-reading-the-documentation)

3. [Reproducing our evaluation](#3-reproducing-our-evaluation)


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

## 2. Reading the documentation

To take a look at the documentation, run the following command from the root of this repository and follow the generated link:

```bash
mkdocs serve
```

## 3. Reproducing our evaluation