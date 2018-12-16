# th-ml-project
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

Tsinghua University - Machine learning final project

## Architecture

Notebook for the project content __only__ visualisation, results or run external script

- _ml_project_: Notebook for project
- _dataset_: Directory where dataset is store (test and train)
- _download_: Script for download data _(no call in notebook)_
- _load_: Script for load data _(call in notebook)_
- _pickles_: Pickles functions (cache function for optimize compute)
- _preprocess_: PreProcess functions
- _script_: Scripts operations directory (run script of, process script and debug script)
- _utils_: Functions utilities (like ThreadPool, ...)
- _visualization_: Function for visualization on notebook

- - - -
## Dependencies
- Python 3.5 >=
- Pip (for python 3.5 >=)
- Python packages describes on _requirements.txt_ file


## Usage

### Install require package
```bash
pip install -r requirements.txt
```

### Run
````bash
python scripts/compute_on_vm.py
````

### Download dataset
#### Train
```bash
python3 download/train.py
```
#### Test
:exclamation: _Dateset is very big (> 6GB)_
```bash
python3 download/test.py
```

### Run Jupyter notebook _(examples)_
```bash
jupyter notebook ml_project.ipynb 
```