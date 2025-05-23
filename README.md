# Submission Branch

## Requirements

### Explanation

`numpy`: Used for numerical computing.

`opencv-python`; Image Processing

`scikit-image`: `hog()` function to extract features

`scikit-learn`: Train classifier apply models.

`pillow`: GUI

`tqdm`: Visualization of training process.

`pandas`: Save result as `.xlsx`

`matplotlib`: Draw DET graphs

### Installation

```bash
pip install -r Others/requirements.txt
```

Use the command to install requirements.

## User Manual

The following is the order of exciting scripts.

1. `trainset_construction.py` 

   Used to construct initial training set.

2. `hog_svm_train.py` 

   Training the first version of model and save in `/models`

3. `hard_neg_constraction.py`

   Add hard examples to training set

4. `retrain.py`

   retrain model with new training set

5. `eval_v1.py` and  `eval_v2.py`

   perform evaluation
