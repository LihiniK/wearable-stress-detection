# Wearable Stress Detection using Physiological Signals and Machine Learning

## Overview

This project develops a machine learning and deep learning pipeline for wearable stress detection using physiological signals from the WESAD dataset.

The goal is to classify physiological signal windows into three affective states:

- Neutral / Baseline
- Stress
- Amusement

The project includes signal preprocessing, window segmentation, feature extraction, classical machine learning, deep learning, Leave-One-Subject-Out evaluation, and an interactive Streamlit app.

---

## Live Demo

An interactive Streamlit application is included for project demonstration.

[Open the Streamlit App](https://wearable-stress-detection-v1.streamlit.app)

The app includes:

- Project overview
- Model comparison dashboard
- Leave-One-Subject-Out evaluation results
- Batch prediction using uploaded feature CSV files
- Prediction probabilities
- Downloadable prediction results

---

## Dataset

This project uses the **WESAD: Wearable Stress and Affect Detection** dataset.

Signals used:

- ECG
- EDA / GSR
- Respiration
- Temperature

The dataset is not included in this repository because of its size. Users should download it manually and place it inside:

```text
data/WESAD/
```

Expected structure:

```text
data/
└── WESAD/
    ├── S2/
    │   └── S2.pkl
    ├── S3/
    │   └── S3.pkl
    └── ...
```

---

## Methodology

### 1. Window Segmentation

Continuous physiological signals were segmented into fixed-size windows.

| Parameter | Value |
|---|---:|
| Sampling rate | 700 Hz |
| Window size | 30 seconds |
| Window size in samples | 21,000 |
| Overlap | 50% |
| Step size | 15 seconds |
| Signals used | ECG, EDA, Respiration, Temperature |

Only the following labels were used:

| Label | Class |
|---:|---|
| 1 | Neutral / Baseline |
| 2 | Stress |
| 3 | Amusement |

Labels `0` and `4` were excluded.

---

### 2. Feature Extraction

For each 30-second window, statistical and signal-change features were extracted.

Feature examples:

- Mean
- Standard deviation
- Minimum and maximum
- Median
- Range
- Interquartile range
- RMS
- Energy
- Mean absolute change
- Zero crossings
- Skewness
- Kurtosis

Final feature dataset:

| Item | Value |
|---|---:|
| Number of windows | 2151 |
| Number of features | 68 |
| Number of subjects | 15 |

---

## Models

### Classical Machine Learning

- Logistic Regression
- Random Forest
- Support Vector Machine
- Gradient Boosting

### Deep Learning

- 1D CNN
- CNN-GRU
- Feature-Based MLP

### Final Deployment Model

The deployed Streamlit app uses:

```text
Gradient Boosting
```

This model was selected because it achieved the best subject-independent performance.

---

## Results

### Single Subject-Independent Train/Test Split

| Model | Type | Accuracy | Macro F1 |
|---|---|---:|---:|
| Gradient Boosting | Classical ML | 0.7067 | 0.5775 |
| 1D CNN | Raw-Signal Deep Learning | 0.3788 | 0.3584 |
| CNN-GRU | Raw-Signal Deep Learning | 0.4642 | 0.3803 |
| Feature-Based MLP | Feature-Based Deep Learning | 0.4642 | 0.4274 |

The best model in the single split experiment was **Gradient Boosting**.

---

### Leave-One-Subject-Out Cross-Validation

| Model | Mean Accuracy | Std Accuracy | Mean Macro F1 | Std Macro F1 |
|---|---:|---:|---:|---:|
| Gradient Boosting | 0.6871 | 0.1638 | 0.5769 | 0.1805 |
| Support Vector Machine | 0.6128 | 0.1976 | 0.5192 | 0.1307 |
| Random Forest | 0.6897 | 0.1128 | 0.5037 | 0.1354 |
| Logistic Regression | 0.5896 | 0.1986 | 0.4948 | 0.1648 |

Best LOSO model:

| Metric | Value |
|---|---:|
| Model | Gradient Boosting |
| Mean Accuracy | 0.6871 |
| Mean Macro F1 | 0.5769 |
| Number of subjects | 15 |
| Number of windows | 2151 |
| Number of features | 68 |

---

## Key Finding

Classical machine learning with handcrafted physiological features outperformed raw-signal deep learning models.

This suggests that feature engineering remains highly effective for small wearable physiological datasets, especially when evaluating generalization to unseen subjects.

---

## Result Visualizations

### All Model Comparison

![All Model Comparison](results/all_model_comparison_macro_f1.png)

### Leave-One-Subject-Out Model Comparison

![LOSO Model Comparison](results/loso_model_comparison_macro_f1.png)

### LOSO Best Model Confusion Matrix

![LOSO Confusion Matrix](results/loso_best_model_confusion_matrix.png)

---

## Streamlit App

The Streamlit app is located in:

```text
app/streamlit_app.py
```

Required app artifacts:

```text
app_artifacts/
├── stress_model.joblib
├── feature_columns.json
├── model_metadata.json
├── feature_input_template.csv
└── example_feature_input.csv
```

Run locally:

```bash
streamlit run app/streamlit_app.py
```

---

## Repository Structure

```text
wearable-stress-detection/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── app/
│   └── streamlit_app.py
│
├── app_artifacts/
│   ├── stress_model.joblib
│   ├── feature_columns.json
│   ├── model_metadata.json
│   ├── feature_input_template.csv
│   └── example_feature_input.csv
│
├── data/
│   ├── WESAD/              # Not uploaded to GitHub
│   └── processed/          # Not uploaded to GitHub
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_window_segmentation.ipynb
│   ├── 03_feature_extraction.ipynb
│   ├── 04_machine_learning_models.ipynb
│   ├── 05_deep_learning_models.ipynb
│   ├── 06_improved_deep_learning_models.ipynb
│   ├── 07_feature_based_neural_network.ipynb
│   ├── 08_leave_one_subject_out_evaluation.ipynb
│   └── 09_export_final_model_for_streamlit.ipynb
│
├── results/
│   └── result CSV files, figures, and reports
│
└── models/
    └── local trained model files
```

---

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/LihiniK/wearable-stress-detection.git
cd wearable-stress-detection
```

### 2. Create a Virtual Environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\activate
```

macOS / Linux:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Download the Dataset

Download the WESAD dataset manually and place it inside:

```text
data/WESAD/
```

### 5. Run the Notebooks

Run notebooks in order:

```text
01_data_exploration.ipynb
02_preprocessing_window_segmentation.ipynb
03_feature_extraction.ipynb
04_machine_learning_models.ipynb
05_deep_learning_models.ipynb
06_improved_deep_learning_models.ipynb
07_feature_based_neural_network.ipynb
08_leave_one_subject_out_evaluation.ipynb
09_export_final_model_for_streamlit.ipynb
```

### 6. Run the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

---

## Important Notes

The following are not uploaded to GitHub:

```text
data/WESAD/
data/processed/
models/*.pkl
models/*.joblib
models/*.h5
models/*.pt
```

The deployed app uses this model artifact:

```text
app_artifacts/stress_model.joblib
```

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Joblib
- PyTorch
- Jupyter Notebook
- Streamlit

---

## Skills Demonstrated

- Physiological signal processing
- Time-series window segmentation
- Statistical feature extraction
- Classical machine learning
- Deep learning for time-series classification
- Subject-independent evaluation
- Leave-One-Subject-Out cross-validation
- Model deployment with Streamlit
- GitHub project organization

---

## Conclusion

This project developed a complete wearable stress detection pipeline using ECG, EDA, respiration, and temperature signals from the WESAD dataset.

The best-performing model was **Gradient Boosting**, achieving a mean LOSO macro F1-score of **0.5769** across 15 subjects.

The results show that handcrafted physiological features are effective for wearable stress classification when the dataset is small and subject-independent generalization is required.

---

## Future Work

Possible future improvements:

- Add wrist sensor signals such as BVP, wrist EDA, wrist temperature, and acceleration
- Add heart-rate variability features from ECG
- Add EDA-specific tonic and phasic features
- Improve deep learning with 1D ResNet or Temporal Convolutional Networks
- Add automated feature extraction inside the Streamlit app
- Evaluate binary classification: stress vs non-stress

---

## Author

**Lihini Karunarathne**

Machine Learning / Deep Learning portfolio project for wearable physiological stress detection.