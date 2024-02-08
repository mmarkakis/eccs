# eccs
Repo for the "Exposing Critical Causal aSsumptions" project.


## Setting up a virtual environment and installing dependencies

Using a virtual environment is recommended to ensure dependencies are managed correctly. This section will walk you through setting up a virtual environment for this project.

### Prerequisites

- Python 3 installed on your system
- Access to the command line/terminal

### Steps

#### 1. Creating the Virtual Environment

First, navigate to the project's root directory in your terminal. Then, create a virtual environment by running:

```bash
python3 -m venv eccs-venv
```

This command creates a new directory `eccs-venv` in your project where the virtual environment files are stored.

#### 2. Activating the Virtual Environment

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

#### 3. Installing Dependencies

With the virtual environment active, install the project dependencies by running:

``` bash
pip install -r requirements.txt
```

#### 4. Deactivating the Virtual Environment
When you're done working in the virtual environment, you can deactivate it by running:

```bash
deactivate
```

This command will return you to your system's default Python interpreter.

