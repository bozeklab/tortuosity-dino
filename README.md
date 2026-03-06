# IVCM DINO
DINO for Tortuosity Grade Classification of Corneal Nerve Fibers.

## 1. Installation
a) Request access to the data from [zenodo](https://zenodo.org/records/12776091). <br>
b) Copy the data to the `data/` directory. <br>
c) In `data/` we provide the `data_split.json` file that contains indices for training/validation/testing splits without the duplicated files. <br> 
To use the subsets without the duplications, rename your data folders to CORN1500_noD and CORN-3_noD. <br>
d) Install dependencies: 
```
pip install -r requirements.txt
```

## 2. Training
For Linear Probing run:
```
python run_ssl_linear_probing.py
```

For Fine-tuning run:
```
python run_sl_dino.py
```