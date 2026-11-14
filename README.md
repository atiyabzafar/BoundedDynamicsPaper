# BoundedDynamicsPaper
 
Code for the paper titled  *Diversity in Well-Stirred Random Replicator Networks* submitted to Front. Appl. Math. Stat.(Currently in review)

`multicoremodel_saving_normal.py` performs the simulations and create the data for the parameters defined in the file

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

- `multicoremodel_saving_normal.py` can be used to create dataset used for dataset used in Figure 1, where $\alpha_{ij}$ are sampled from a normal distribution. around 1.
- `multicoremodel_saving_normal.py` can be used for data used in Figure 2,3,4 where $\alpha_{ij}=1$ for all.

Edit the parameters defined in the code to save the result. It outputs a CSV file with diversity, seed and other values for analysis and saves it in the root directory.

## Publication figures

Publication figures are made using the code in the jupyter notebook `publication_figures.ipynb`. It uses CSV files saved in `output/publication`. If you wish to point to new data collected using your own simulations, please rename the files appropriately. Do not delete `output/publication` directory. 
