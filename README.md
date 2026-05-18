# IVCM DINO
DINO for Tortuosity Grade Classification of Corneal Nerve Fibers.

## 1. Installation
a) Request access to the data from [zenodo](https://zenodo.org/records/12776091). <br>
b) Copy the data to the `data/` directory. <br>
c) In `data/` we provide the `data_split.json` file that contains indices for training/validation/testing splits without the duplicated files. <br> 
To use the subsets without the duplications, rename your data folders to CORN1500_noD and CORN-3_noD. <br>
d) Install dependencies: <br>
Experiments were run on Python 3.12.12 <br>
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

## Citation

If you used this work, please cite:

```
@misc{ouan_self-supervised_2026,
	title = {Self-{Supervised} {ImageNet} {Representations} for {In} {Vivo} {Confocal} {Microscopy}: {Tortuosity} {Grading} without {Segmentation} {Maps}},
	url = {http://arxiv.org/abs/2603.15269},
	doi = {10.48550/arXiv.2603.15269},
	urldate = {2026-05-18},
	publisher = {arXiv},
	author = {Ouan, Kim and Moreau, Noémie and Bozek, Katarzyna},
	month = may,
	year = {2026},
	note = {arXiv:2603.15269 [cs]},
	keywords = {Computer Science - Computer Vision and Pattern Recognition},
}
```
