# Dataset Information

This project uses the **Onion Dataset Combined** collection for four-class onion-condition classification. Raw images are not included in this repository.

## Dataset Link

Google Drive Link: **Not provided with the project**

Official Kaggle Dataset:  
[Onion Dataset Combined](https://www.kaggle.com/datasets/yashraneja2/onion-dataset-combined)

Download the dataset directly from its official Kaggle page and follow the dataset owner's license and Kaggle's terms of use. Do not upload raw dataset images, credentials, Kaggle API tokens, or privately shared copies to this repository.

## Dataset Statistics

The statistics below were recovered from executed outputs in the supplied training notebooks. The source dataset is divided into fixed `train`, `valid`, and `test` directories.

### Before Augmentation

| Class | Training Images | Validation Images | Test Images | Total Images |
| --- | ---: | ---: | ---: | ---: |
| Molded (`molded`) | 264 | 75 | 38 | 377 |
| Normal (`normal`) | 708 | 202 | 101 | 1,011 |
| Rotten (`rotten`) | 149 | 43 | 21 | 213 |
| Sprouted (`sprouted`) | 651 | 186 | 93 | 930 |
| **Total** | **1,772** | **506** | **253** | **2,531** |

Average Images per Class:

- Training set: **443**
- Validation set: **126.5**
- Test set: **63.25**
- Complete dataset: **632.75**

The source data is class-imbalanced. The `normal` and `sprouted` classes contain substantially more images than the `molded` and `rotten` classes. The supplied training workflows address this imbalance using class weighting, Macro-F1 monitoring, balanced-accuracy monitoring, and optional focal loss.

### After Augmentation

Training augmentation is performed **dynamically in memory** for each batch. It does not create or save a fixed expanded dataset. Consequently, there is no authoritative after-augmentation image count.

| Class | Stored Images | Effective Images After Augmentation |
| --- | ---: | --- |
| Molded (`molded`) | 377 | Generated dynamically; no fixed total |
| Normal (`normal`) | 1,011 | Generated dynamically; no fixed total |
| Rotten (`rotten`) | 213 | Generated dynamically; no fixed total |
| Sprouted (`sprouted`) | 930 | Generated dynamically; no fixed total |
| **Total** | **2,531** | **Not fixed** |

Average stored images per class: **632.75**  
Average effective augmented images per class: **Not fixed**

Only training images are augmented. Validation and test images remain unaugmented to provide stable and comparable evaluation results. The training pipeline uses condition-preserving transformations and avoids extreme color changes that could hide or alter visible surface symptoms.

## Class Description

### Molded (`molded`)

Onions showing visible mold-related deterioration, such as fungal growth, mold patches, abnormal surface discoloration, or related spoilage. The dataset folder and model label use the exact identifier `molded`; the application may display the friendlier term **Moldy**.

### Normal (`normal`)

Onions without the targeted visible signs of mold, rot, or sprouting. These samples represent acceptable-looking onions according to the dataset's labeling policy.

### Rotten (`rotten`)

Onions showing visible decay or decomposition. Possible indicators include darkened regions, damaged tissue, collapsed or softened surfaces, and severe discoloration.

### Sprouted (`sprouted`)

Onions showing visible sprout growth, including emerging green shoots, developed stems, or other clear signs that the bulb has begun germinating.

## Dataset Usage Guidelines

- Preserve the exact class order expected by the selected trained model.
- Keep `molded` and `rotten` as separate labels.
- Keep validation and test images outside the training pipeline to prevent data leakage.
- Check for duplicate and near-duplicate images across splits before retraining.
- Treat the dataset owner's original annotations and labeling policy as authoritative.
- Obtain any necessary permission before redistributing images or derived dataset archives.

Return to the [main README](../README.md).
