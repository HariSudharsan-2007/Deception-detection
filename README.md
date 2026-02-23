# EEG-Based Deception Detection System

## Overview
This project presents a Brain-Computer Interface (BCI) machine learning pipeline designed to detect deception using electroencephalogram (EEG) signals. By analyzing cognitive responses to visual stimuli, the system classifies whether a subject is acting as a truth-teller or a deceiver. The project implements both gradient boosting and recurrent neural network architectures to evaluate the temporal dependencies inherent in EEG data.

## Dataset
The project utilizes the **LieWaves Dataset**, which consists of EEG recordings from 27 subjects. Data was acquired using a 5-channel Emotiv Insight wearable EEG device.

### Experimental Paradigm
* **Task**: Subjects were presented with a box of 5 different beads and instructed to place 2 in their pockets.
* **Stimuli**: Subjects watched a video displaying images of the beads (alternating 2 seconds of bead images and 1 second of a black screen).
* **Roles**: Subjects participated in two distinct experiments, alternating between a "deceiver" role (answering falsely to all prompts via left/right button clicks) and a "truth-teller" role (answering truthfully).

### Data Structure
The repository contains the following data variations to facilitate comparative model performance:
* `Raw`: Unprocessed 75-second continuous EEG trials in `.csv` format.
* `1_BandPass_Filtered`: Data processed with a 0.5-45 Hz bandpass filter.
* `2_ASR`: Data cleaned using Artifact Subspace Reconstruction.
* `3_ICA`: Data processed via Independent Component Analysis.
* `4_ATAR`: Data filtered using Automatic and Tunable Artifact Removal.

## Methodology and Code Implementation

### 1. Data Preprocessing
The time-series EEG data was trimmed to remove the initial 2 seconds of excessive noise at the start of each trial, resulting in standardized 75-second recording windows for feature extraction.

### 2. Model Architecture
The project explores multiple classification algorithms to handle the sequential nature of the EEG data:
* **XGBoost**: Utilized as a baseline ensemble classifier for rapid evaluation of the extracted features.
* **Long Short-Term Memory (LSTM)**: A recurrent neural network architecture implemented to map the temporal sequences of the brainwave data. A sigmoid activation threshold of 0.5 is used on the flattened predictions to output the final binary class.

### 3. Training and Evaluation
The models are evaluated using standard classification metrics:
* **Metrics**: The pipeline outputs the global accuracy score, alongside a detailed classification report encompassing precision, recall, and f1-score for both classes.
* **Visualizations**: 
    * Confusion matrices are generated using `seaborn.heatmap` to visualize True Positive and False Positive distributions across the truth and deception classes.
    * Training and validation loss curves are plotted across epochs to monitor model convergence and detect overfitting in the LSTM network.

## Dependencies
* Python 3.x
* pandas
* scikit-learn
* xgboost
* TensorFlow / Keras (for LSTM architecture)
* matplotlib
* seaborn
