# **NINCO** dataset utilities
### and example notebooks for replicating the results of the paper 
# "In or Out? Fixing ImageNet Out-of-Distribution Detection Evaluation"

This repository contains the evaluation pipeline for obtaining the results in the paper.
It was tested with the python package versions that are installed via `pip install -r requirements.txt`.

Five jupyter notebooks can in the current version be used as tutorials for the repo's functionalities.

The file `model_outputs/df.pkl` contains most evaluation results shown in the paper and is opened in `plots_in_or_out_paper.ipynb` -- these results can be accessed without downloading data or running models.

---

## NINCO Dataset Download 
[To evaluate models and view the NINCO images, please download and extract this tar.gz file.](https://drive.google.com/file/d/1lR9ncSCyLH6uVb4jzfZtMRPg1YMiYQGt/view?usp=share_link)

--- 

Please edit `data/paths_config.py` to set `ninco_folder` to the folder where the downloaded datasets have been extracted (containing the folders `NINCO_OOD_classes`,  `NINCO_OOD_unit_tests` and  `NINCO_popular_datasets_subsamples`).
Also, set `repo_path` to the absolute path of this repository.

Models can be evaluated by running `evaluate.py`.

For example, the ViT-B-384-21k with MSP can be evaluated on the NINCO OOD classes with:

`python evaluate.py --model_name vit_base_patch16_384_21kpre --dataset NINCO --method MSP`

Specifying `--method all` evaluates all methods, but takes considerable time since it requires a forward pass over the whole ImageNet train set.

Example methods for examining results of such evaluations are shown in `analyze_evaluations.ipynb`.