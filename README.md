# TRAFFIC SIGN CLASSIFICATION – COMP30027 PROJECT 2 (2025)

## Overview:
This project aims to classify German traffic signs into one of 43 classes using extracted features. 

## Directory Structure:
The project is organized as follows:
```
|- train/                        # Folder containing all training images and related files
|    |- train_metadata.csv       # CSV file with metadata for training images (IDs, filenames, and class labels)
|    |- Features/                # Folder containing extracted features for training images
|         |- color_histogram.csv     # Color histogram features for each training image
|         |- hog_pca.csv             # HOG (Histogram of Oriented Gradients) features reduced by PCA for each training image
|         |- additional_features.csv # Any additional hand-crafted features for training images
|         |- shape_features.csv      # Shape-related features for each training image
|         |- texture_features.csv    # Texture-related features for each training image
|- test/                         # Folder containing all test images and related files
|    |- test_metadata.csv        # CSV file with metadata for test images (IDs and filenames, no labels)
|    |- Features/                # Folder containing extracted features for test images
|         |- color_histogram.csv
|         |- hog_pca.csv
|         |- additional_features.csv
|         |- shape_features.csv
|         |- texture_features.csv
|- processed_data/               # Folder containing processed and feature-selected datasets for training, validation, and test data
|    |- train_rf                    # Training set after Random Forest feature selection
|    |- train_lasso                 # Training set after LASSO feature selection
|    |- train_mi                    # Training set after Mutual Information feature selection
|    |- val_rf                      # Validation set after Random Forest feature selection
|    |- ...                         # Other processed/validation/test sets for different feature selection methods
|- feature_extraction.ipynb      # Extracting features from raw data
|- feature_processing.ipynb      # Processing, scaling, and selecting features
|- classifier_trainiing.ipynb    # Training traditional ML classifiers
|- hyperparameter_tuning.ipynb   # Hyperparameter tuning on best-performing model
|- traffic-sign-classification-cnn.ipynb # CNN model training and evaluation
|- README.txt                    # This file
```

## Data:
- 5488 training images with class labels
- 2353 test images without labels
- 43 total traffic sign classes
- Provided features:
    * HOG (Histogram of Oriented Gradients) - PCA reduced
    * Color histograms
    * Additional features (edge density, texture variance, mean RGB)

## Submission Format (Kaggle) (CSV):
Final submission should follow this structure:

| id | ClassId |
| -- | ------- |
| 67 | 4 |
| 94 | 2 |
| ... | ... |
| 521 | 12 |

## Python Version:
`3.9.19`

## Requirments:
- numpy
- pandas
- opencv
- tqdm
- pillow
- scikit-learn
- scikit-image

## Run Order
To reproduce the workflow, run the following notebooks in order:
1. feature_extraction.ipynb
2. feature_processing.ipynb
3. classifier_trainiing.ipynb
4. hyperparameter_tuning.ipynb

The CNN model defined in `traffic-sign-classification-cnn.ipynb` was trained using accelerator GPU T4*2 on Kaggle.

The dataset used for training and testing was downloaded from the LMS platform and uploaded to Kaggle manually. The folder structure is as follows:

```
2025_A2/
|- train/
|   |- img_000001.jpg
|   |- ...
|- test/
    |- img_005489.jpg
    |- ...
```

The best-performing model and the corresponding test prediction results are saved in the `/kaggle/working/` directory.
