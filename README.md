
# ML-Based Signal Detection

## Project Summary

This project implements a machine learning-based approach to detect the presence or absence of a signal in noisy observations. It compares a Random Forest classifier with a conventional energy detector to study their detection performance under different noise conditions.

## LiveDemo
Experience the application here:  
https://ml-signal-detection-vbjmxaydt3v7wnhn7nbxck.streamlit.app/

## Objectives

- Detect whether a signal is present in a noisy observation.
- Implement a conventional energy-based detection method.
- Train a Random Forest classifier using features extracted from noisy signals.
- Compare the performance of both detectors using evaluation metrics.
- Study how increasing noise levels affect signal detection performance.
- Visualize noisy observations and model feature importance through an interactive dashboard.

## Features

- **Signal Generation:** Generates synthetic sinusoidal signals corrupted by additive white Gaussian noise (AWGN).
- **Energy Detector:** Classifies observations by comparing their mean-square energy against a calibrated threshold.
- **Random Forest Classifier:** Uses extracted signal features to classify observations as signal-present or signal-absent.
- **Performance Evaluation:** Displays accuracy, precision, detection probability, false-alarm rate, miss rate, and F1 score.
- **Confusion Matrices:** Visualizes correct and incorrect classifications for both detectors.
- **Noisy Signal Visualization:** Displays example observations in both time and frequency domains.
- **Feature Importance:** Shows the relative importance of features used by the trained Random Forest.
- **Noise-Level Comparison:** Evaluates and compares detection probability across different noise levels.
- **Interactive Dashboard:** Allows users to modify evaluation settings through Streamlit's sidebar.

## Tech Stack
- **Language**: Python
- **Libraries**: NumPy, Pandas, Scikit-learn, Matplotlib
- **Techniques**: Signal Processing, Feature Extraction, Energy-Based Detection
- **Models**: Random Forest Classifier, Energy Detector
- **Frontend**: Streamlit
- **Noise Model**: Additive White Gaussian Noise (AWGN)
