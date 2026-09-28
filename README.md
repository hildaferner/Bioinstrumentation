# EEG Eye-State Classification (Eyes Closed vs. Eyes Open)

A machine learning pipeline to classify human physiological states (eyes closed vs. eyes open) from 64-channel electroencephalogram (EEG) recordings.

Developed as a course project for Bioinstrumentation at Université catholique de Louvain (UCLouvain).

---

## Project Overview

The objective of this project is to train a machine learning model capable of distinguishing whether a subject's eyes are open or closed based on short (500 ms) segments of multi-channel EEG signals.

The task leverages the Berger effect, the marked desynchronization and attenuation of occipital Alpha band power (8–13 Hz) upon visual stimulation (eyes opening), in conjunction with frontal oculomotor artifacts (blinks and saccades).

---

## Processing Pipeline
```text
Raw EEG Recording (64 channels, 1024 Hz)
   │
   ▼
Initial Quality Inspection & Spectrum Profiling (MNE-Python)
   │
   ▼
Band-Pass Filtering (0.5 – 40 Hz, Zero-Phase FIR)
   │
   ▼
Post-Filter Quality Assessment (Variance Z-Score Profiling)
   │
   ▼
Bad Channel Removal (Persistent outlier AF7 discarded)
   │
   ▼
Segmentation: 500 ms Epochs (50% Overlap, 250 ms Step)
   │
   ▼
Segment Artifact Rejection (Exclusion of ~63 s contamination region)
   │
   ▼
Feature Extraction (Relative Band Power: Delta, Theta, Alpha, Beta via Welch PSD)
   │
   ▼
Temporal Block-Wise Train/Test Split (80/20)
   │
   ▼
Dimensionality Reduction (PCA)
   │
   ▼
Model Training & Hyperparameter Tuning (5-Fold Cross-Validation)
   │
   ▼
Performance Evaluation (Accuracy, F1-Score, Confusion Matrix)
```
---

## Methods and Implementation

* **Signal Handling and Visualization:** Conducted using `MNE-Python` for raw trace inspection, 10–10 system sensor mapping, and Power Spectral Density (PSD) analysis.
* **Filtering:** Zero-phase FIR band-pass filter from 0.5 to 40 Hz to suppress low-frequency DC drift, movement artifacts, and 50 Hz powerline interference.
* **Channel Selection:** Quantitative variance analysis (|Z| > 3) detected AF7 as a persistent noise outlier, which was systematically removed across conditions to maintain a consistent topology.
* **Epoching and Artifact Cleaning:** 
  * Continuous data divided into 500 ms windows with 50% overlap (250 ms step).
  * High-amplitude contamination occurring around the 63-second mark was discarded prior to downstream analysis.
* **Feature Engineering:** Calculated relative band power using Welch's PSD normalized over the 1–40 Hz broadband spectrum:
  * **Delta:** 1–4 Hz
  * **Theta:** 4–8 Hz
  * **Alpha:** 8–13 Hz (primary marker of eye closure)
  * **Beta:** 13–30 Hz
* **Data Partitioning:** Applied an 80/20 train/test split along contiguous temporal blocks to prevent data leakage between overlapping windows.
* **Dimensionality Reduction:** Principal Component Analysis (PCA) applied to reduce collinearity across spatial channels and mitigate the curse of dimensionality.
* **Classification and Evaluation:** Benchmarked classifiers with hyperparameter optimization using 5-Fold Cross-Validation on the training partition.

---

## Results Summary

Overall results confirm that EEG spectral features provide strong discriminative power for distinguishing eye states. The relative power of the occipital Alpha band (8–13 Hz) forms the primary physiological separation boundary, supported by frontal variance differences from ocular activity. PCA effectively compressed multi-sensor redundancy while preserving the variance necessary to achieve robust binary classification on the test set.

---
