# Reduced Data Set


The publicly available data sets $\mathrm{CORN}^{1500}$ 
and CORN-3 offered great value to our work. 
We want to emphasize that our published results were achieved under 
the same experimental setup as Mou et al. (2022) (DeepGrading: Deep Learning Grading of Corneal Nerve Tortuosity)
to ensure comparability. 
However, during our work on the data sets,
we noticed the following constraints:

- $\mathrm{CORN}^{1500}$ and CORN-3 contain duplicated images.
- A few images of $\mathrm{CORN}^{1500}$ are also in CORN-3. 
Thus, there is a data leakage between training, validation and test set.
- Duplicated images can have up to three different labels.
- We downloaded a new version of CORN-3 on 15.12.2025 from the official publication 
of the data set on zenodo.org which was a reduced set (242 samples) of the original 
(403 samples), which we will call CORN-3\_v2.
- CORN-3\_v2 also contains duplicated images.

These limitations create an unintended emphasis on selected samples during training 
as well as a data leakage between the training and the test set. 
We implement the following measurements to resolve the aforementioned constraints:

- Remove duplicated files in $\mathrm{CORN}^{1500}$.
- Remove duplicated files in CORN-3\_v2.
- Remove duplicated files between CORN-3\_v2 and $\mathrm{CORN}^{1500}$ from CORN-3\_v2.

While removing the duplicated images, we use majority voting for the labels and 
remove samples for which there is a tie in the majority vote, as the label for 
these samples would be unclear. We rename our reduced data sets to 
$\mathrm{CORN}^{1500}\text{-noD}$ and CORN-3-noD. $\mathrm{CORN}^{1500}\text{-noD}$ 
contains 1250 images distributed across the four levels: 188, 396, 289, 377, respectively. 
CORN-3-noD contains 199 images distributed across the four levels: 28, 91, 65, 15, 
respectively. We retrain our model on $\mathrm{CORN}^{1500}\text{-noD}$ with a 
data split of 70/30 for training and validation and evaluated it on CORN-3-noD. 
The final results on CORN-3-noD are shown below. All used data samples can be found in
our [data_split.json](data_split.json).

Methods | Metrics | level1 | level2 | level3 | level4 | overall
--- | --- | --- | --- | --- | --- | ---
DINO ViT-B/16 (finetuned) | $wAcc$ | 86.93 | 76.38 | 88.94 | 99.50 | 83.71
"| $wSe$ | 85.71 | 73.63 | 69.23 | 100.00 | 75.88
" | $wSp$ | 87.13 | 78.70 | 98.51 | 99.46 | 87.92
