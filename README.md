# Wearable Stress Detection using Physiological Signals and Machine Learning

## Overview

This project develops a machine learning pipeline for stress detection using wearable physiological sensor data. The goal is to classify affective states such as neutral, stress, and amusement from signals such as ECG, EDA/GSR, respiration, temperature, and acceleration.

## Dataset

This project uses the WESAD dataset: Wearable Stress and Affect Detection. The dataset contains physiological and motion sensor data collected from subjects during a laboratory stress-affect study.

The dataset is not included in this repository due to its size. Users should download it manually from the official WESAD dataset source.

## Project Goals

- Load and explore wearable physiological sensor data
- Visualize ECG, EDA/GSR, respiration, and temperature signals
- Segment time-series signals into fixed-size windows
- Extract statistical and physiological features
- Train machine learning models for stress classification
- Evaluate models using accuracy, macro F1-score, and confusion matrix
- Prepare the project for GitHub and CV/portfolio use

## Methods

Planned methods:

- Signal preprocessing
- Window-based segmentation
- Feature extraction
- Random Forest
- Support Vector Machine
- Gradient Boosting
- Deep learning model in later stage

## Repository Structure

```text
wearable-stress-detection-ml-dl/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
├── notebooks/
├── src/
├── results/
├── models/
└── app/