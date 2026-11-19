# PhishJob: Machine Learning-Based URL Phishing Detection

This repository contains the source code, experimental notebooks, datasets, trained model artifacts, and deployment implementation for the research entitled:

**"A Comparative Analysis of Machine Learning Algorithms for URL Phishing Detection in Job Recruitment."**

The project develops a two-stage URL phishing detection system using lexical and structural URL features. The first stage applies supervised machine learning algorithms to classify URLs as legitimate or phishing. The second stage uses a Hybrid Self-Attention Autoencoder and K-Means clustering to explore structural patterns among URLs classified as phishing.

## 1. Research Overview

The proposed framework consists of two main stages.

### Stage 1: Supervised URL Classification

Five machine learning algorithms are investigated and compared:

- Logistic Regression (LR)
- Support Vector Machine (SVM)
- Random Forest (RF)
- XGBoost
- LightGBM

URL lexical and structural characteristics are extracted and used as input features for classification. Hyperparameter optimization is performed to identify suitable model configurations.

Based on the experimental comparison, LightGBM is selected as the best-performing classification model and used in the deployed application.

### Stage 2: Exploratory URL Clustering

URLs classified as phishing by the Stage 1 LightGBM model are further analyzed using:

1. Hybrid Self-Attention Autoencoder for latent representation learning.
2. K-Means for clustering the resulting latent representations.

This stage aims to investigate latent structural patterns and similarities among predicted phishing URLs. The resulting clusters are exploratory and should not be interpreted as independently verified phishing categories.

## 2. Repository Structure

The repository is organized as follows:

```text

Phishjob/
│
├── data/
│   ├── testing_features_24.csv
│   └── training_features_24.csv
│
├── model/
│   ├── LOGREG.ipynb
│   ├── Paper_ICDSOS_Hybrid_Multimodal_Pipeline_v2_CV7model.ipynb
│   ├── phishing_rf_colab (3).ipynb
│   └── svm_classification_phising_cv_fixed (1).ipynb
│
├── server testing/
│   ├── benchmark_20_urls.py
│   └── load_test.py
│
├── stage2_export/
│   ├── ae_hybrid_s0.pt
│   ├── stage2_meta.json
│   ├── stage2_params.npz
│   └── tokenizer_bpe.json
│
├── app.py
├── phishjob_features.py
├── phishjob_lgbm.txt
├── phishjob_lgbm_meta.json
├── phishjob_stage2.py
├── pipeline_data_prepocessing_extractfeatures.py
├── requirements.txt
└── README.md

```

## 3. Directory and File Descriptions

### 3.1 Data

The `data/` directory contains the processed datasets used in the feature-based classification pipeline.

| File                       | Description                                             |
| -------------------------- | ------------------------------------------------------- |
| `training_features_24.csv` | Training dataset containing the extracted URL features. |
| `testing_features_24.csv`  | Testing dataset containing the extracted URL features.  |

The datasets contain 24 extracted URL features used in the research pipeline.

### 3.2 Model Experiments

The `model/` directory contains Jupyter notebooks documenting the experimental implementation and evaluation of the Stage 1 classification methods.

The notebooks include:

- Logistic Regression experiments.
- Support Vector Machine classification.
- Random Forest classification.
- Research implementation and hybrid model experiments.

These notebooks provide supporting documentation for the model development and experimental comparison.

### 3.3 Server Testing

The `server testing/` directory contains scripts for evaluating the performance of the deployed application.

| File                   | Description                                                                  |
| ---------------------- | ---------------------------------------------------------------------------- |
| `benchmark_20_urls.py` | Sequential benchmarking of server-side processing time using 20 URLs.        |
| `load_test.py`         | Script for evaluating application performance under load-testing conditions. |

The benchmarking process measures the processing time associated with feature extraction, prediction, and clustering, where applicable.

### 3.4 Stage 2 Export

The `stage2_export/` directory contains the exported model artifacts required for Stage 2 clustering inference.

| File                 | Description                                             |
| -------------------- | ------------------------------------------------------- |
| `ae_hybrid_s0.pt`    | Trained Hybrid Self-Attention Autoencoder weights.      |
| `stage2_meta.json`   | Model metadata and configuration.                       |
| `stage2_params.npz`  | Numerical parameters required for clustering inference. |
| `tokenizer_bpe.json` | Byte-Pair Encoding tokenizer configuration.             |

### 3.5 Main Application Files

The following files are located in the repository root:

| File                                        | Description                                                                 |
| ------------------------------------------- | --------------------------------------------------------------------------- |
| `app.py`                                    | Main Shiny for Python application for URL phishing detection.               |
| `phishjob_features.py`                      | URL normalization and lexical/structural feature extraction implementation. |
| `phishjob_lgbm.txt`                         | Exported LightGBM classification model used by the application.             |
| `phishjob_lgbm_meta.json`                   | Metadata and configuration for the deployed LightGBM model.                 |
| `phishjob_stage2.py`                        | Stage 2 model implementation and clustering inference.                      |
| `pipeline_data_preprocessing_extraction.py` | Data preprocessing and URL feature extraction pipeline.                     |
| `requirements.txt`                          | Python dependencies required by the application.                            |

## 4. Dataset Sources

The research uses publicly available URL sources:

**Legitimate URLs**

- Source: [Tranco](https://tranco-list.eu/)
- Used to obtain legitimate website URLs.

**Phishing URLs**

- Source: [PhishTank](https://phishtank.org/)
- Used to obtain verified phishing URL records.

The collected URLs undergo preprocessing, including URL cleaning, duplicate handling, and feature extraction.

The processed feature datasets are provided in the `data/` directory. Researchers should refer to the original sources for dataset acquisition and applicable usage conditions.

## 5. Feature Extraction

The system uses lexical and structural characteristics of URLs rather than accessing the actual webpage content.

The feature extraction pipeline initially generates 24 URL features. Following feature selection and preprocessing, the classification experiments use the corresponding feature sets according to the requirements of each algorithm.

The feature extraction implementation used by the deployed application is provided in `phishjob_features.py`.

## 6. Installation

Clone the repository:

```bash
git clone https://github.com/IlhamTessa13/Phishjob.git
cd Phishjob
```

Create a Python virtual environment:

```bash
python -m venv venv
```

Activate the environment.

**Windows:**

```bash
venv\Scripts\activate
```

**Linux/macOS:**

```bash
source venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## 7. Running the Application

The web application is implemented using Shiny for Python.

Run the application locally using:

```bash
shiny run --reload app.py
```

The application provides an interface for submitting URLs and obtaining:

- URL classification results (Legitimate or Phishing).
- Model prediction confidence.
- Exploratory clustering results for URLs classified as phishing.

The deployed application is accessible through Posit Connect Cloud:

**Application URL:**  
https://01a18820-2551-2ec0-1a87-dd2045e6a2df.share.connect.posit.cloud/

## 8. Server Performance Evaluation

Server-side performance is evaluated using the scripts provided in the `server testing/` directory.

The benchmarking procedure measures the processing time of the following stages:

1. Feature extraction.
2. LightGBM prediction.
3. Clustering, when applicable.
4. Total server-side processing time.

The benchmark uses 20 URLs submitted sequentially to the deployed application. Processing-time statistics include the mean, median, 95th percentile, and maximum.

## 9. Reproducibility

This repository provides the main implementation files, experimental notebooks, processed feature datasets, and exported model artifacts used in the proposed system.

The repository is intended to support inspection of the feature extraction pipeline, classification implementation, Stage 2 clustering inference, and deployed application.

The original datasets can be obtained from their respective public sources. Reproducing the complete experimental results may require the original dataset snapshots and the corresponding training configurations.

## 10. Limitations

The proposed system has several limitations:

1. The classification model identifies general phishing URLs and does not exclusively identify job recruitment phishing.
2. Stage 2 clustering is exploratory and does not provide independently verified semantic categories.
3. The model relies on lexical and structural URL characteristics without analyzing webpage content.
4. Classification performance may vary when applied to previously unseen domains and changing phishing patterns.
