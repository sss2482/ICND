# ICND Package

This repository contains the ICND package with various experimentation scripts for \[briefly describe purpose, e.g., data analysis, machine learning, etc.\].

## Cloning the Repository

To get started, clone the repository to your local machine:

```bash
git clone https://github.com/sss2482/ICND.git
```

## Experimentation Scripts

You will find different experimentation scripts in the `mains` folder. Each script is designed to run different types of experimentation.

To store the results of the experiments, specify the path in the `save_name` attribute of the `visualise_counts` function within each script in the `mains` folder.

## Running the Scripts

To run a specific script, use the following command from the root directory of the repository:

```bash
python -m ICND.mains.[file_name]
```

Replace `[file_name]` with the name of the script you want to run (without the `.py` extension). For example:

```bash
python -m ICND.mains.did
```

## Requirements

Ensure you have Python installed. Install any required dependencies by running:

```bash
pip install -r requirements.txt
```


