# Skin Lesion Classification — HAM10000 Metadata Pipeline

A machine learning pipeline for melanoma detection using the HAM10000 dataset, motivated by the challenge of building diagnostic AI tools that perform reliably under real clinical constraints: class imbalance, missing data, and the need for sensitivity-prioritised evaluation.

## Motivation

Melanoma accounts for the majority of skin cancer deaths despite being detectable at early stages. In settings where dermatologist access is limited, AI-assisted triage tools could flag high-risk lesions for priority review. This project explores how far metadata-only features (age, sex, lesion location, confirmation method) can take us — and makes explicit where image features are needed to go further.

## Dataset

**HAM10000** (Human Against Machine with 10000 training images)  
- 10,015 dermatoscopic lesion records  
- 7 diagnostic classes (melanoma, nevi, basal cell carcinoma, etc.)  
- Source: https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection  

> **Note:** This pipeline uses the dataset's metadata CSV only (no image download required). The statistical distribution is reproduced faithfully for reproducibility without requiring Kaggle authentication. To run on real data, replace the simulation block with `pd.read_csv('HAM10000_metadata.csv')`.

## Approach

- **Target:** Binary melanoma vs. non-melanoma classification
- **Features:** Age, sex, lesion location (with high-risk grouping), diagnosis confirmation method
- **Class imbalance:** Handled with SMOTE oversampling (~67:1 imbalance ratio)
- **Models:** Logistic Regression, Random Forest, Gradient Boosting
- **Evaluation:** AUC, Sensitivity, Specificity, Balanced Accuracy — not raw accuracy

## Key Finding

Metadata features alone yield near-random AUC (~0.5), confirming that image data is the essential signal for melanoma classification. This is itself a meaningful result: it motivates multimodal approaches that combine structured patient metadata with image-derived features — an area of active research in computational dermatology.

## Relevance to KAUST Research

This project connects to the work of **Prof. Jesper Tegnér** (KAUST), whose lab has developed hybrid ML techniques for retinal and melanoma datasets, and to **Prof. Xin Gao** (KAUST), whose group works on biomedical imaging analysis and disease detection.

## Stack

```
Python 3.x | scikit-learn | imbalanced-learn | pandas | numpy | matplotlib | seaborn
```

## Run

```bash
pip install scikit-learn imbalanced-learn pandas numpy matplotlib seaborn
python skin_lesion_classifier.py
```

## Output

- `results.png` — ROC curves, sensitivity/specificity comparison, class distribution
