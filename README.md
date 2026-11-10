# BoundedDynamicsPaper
 
Code for the paper titled  *Diversity in Well-Stirred Random Replicator Networks* submitted to Front. Appl. Math. Stat.(Currently in review)

`multicoremodel_saving.py` performs the simulations and create the data for the parameters defined in the file

## Setup

Create and activate a virtual environment, then install the required packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib scipy
```

The `csv`, `concurrent.futures`, and `multiprocessing` modules are included with Python and do not need to be installed separately.

## Run the model

Edit the parameters defined in the code to save the result. It outputs a CSV file with diversity, seed and other values for analysis. 
