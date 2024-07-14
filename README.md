# Music Equipment Detection from Audio
This repository contains a deep learning project aimed at detecting and classifying various music instruments from audio recordings. The project uses the librosa library for audio feature extraction and a neural network built with TensorFlow/Keras for classification. Additionally, hyperparameter tuning is performed using Keras Tuner to optimize the model.

### Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Model](#model)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [Results](#results)
7. [Usage](#usage)
8. [Dependencies](#dependencies)
9. [License](#license)

### Introduction

The purpose of this project is to accurately classify 28 different musical instruments from audio recordings. The project involves the following steps:

1. Loading and preprocessing the audio data.
2. Extracting MFCC features.
3. Data augmentation to improve model robustness.
4. Building and tuning a neural network model.
5. Training the model and evaluating its performance.
6. Visualizing the results and generating a classification report.

### Dataset

To use this project, you will need a dataset of audio recordings of various musical instruments. 

**Note:** The dataset is not included in this repository. You can use a publicly available dataset or your own collection of audio files. Ensure that your dataset is organized in the following structure:

```
datasets/
├── Instrument1/
│   ├── file1.wav
│   ├── file2.wav
│   └── ...
├── Instrument2/
│   ├── file1.wav
│   ├── file2.wav
│   └── ...
└── ...
```

### Model

The model is a feedforward neural network with three dense layers. Hyperparameter tuning is performed using Keras Tuner to find the optimal number of units, dropout rates, and the optimizer.

### Training

The training script performs the following steps:

1. Load and preprocess the data.
2. Split the data into training and testing sets.
3. Perform hyperparameter tuning.
4. Train the model with the best hyperparameters.
5. Evaluate the model on the test set.

### Evaluation

The evaluation includes the following:

- Accuracy and loss plots.
- Confusion matrix.
- Classification report with precision, recall, and F1-score.

### Results

The trained model achieved a test accuracy of 97.90%. Detailed results and performance metrics are provided in the evaluation section.

### Usage

To run the project, follow these steps:

1. Clone the repository:

```
git clone https://github.com/abdulvahapmutlu/music-equipment-detection.git
cd music-equipment-detection
```

2. Install the dependencies:

```
pip install -r requirements.txt
```

3. Place your dataset in the `datasets` directory.

4. Run the training script:

```
python train.py
```

### Dependencies

- Python 3.x
- librosa
- numpy
- pandas
- scikit-learn
- tensorflow
- keras-tuner
- matplotlib
- seaborn

Install the dependencies using:

```
pip install -r requirements.txt
```


### License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
