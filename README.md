## GitHub Repository Description

### Project: Emotion Detection using Capsule Networks on EEG Data

This repository contains the implementation of a Capsule Network (CapsNet) for emotion detection based on EEG (Electroencephalogram) data. The approach utilizes the Capsule Network architecture to classify emotions into three categories: NEGATIVE, NEUTRAL, and POSITIVE.

### Dataset Details

- **EEG Brainwave Dataset (Feeling Emotions)**: The dataset used is sourced from Kaggle and contains EEG recordings labeled with different emotions. The dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-feeling-emotions).

### Tech Stack

- **Programming Language**: Python
- **Libraries and Tools**:
  - `pandas`: For data manipulation and analysis.
  - `numpy`: For numerical computations.
  - `matplotlib` and `seaborn`: For data visualization.
  - `plotly`: For interactive visualizations.
  - `scipy`: For signal processing.
  - `scikit-learn`: For data preprocessing and evaluation metrics.
  - `tensorflow` and `keras`: For building and training the Capsule Network model.
  - `opendatasets`: For downloading datasets from Kaggle.

### Algorithm and Approach

1. **Environment Setup**: Install the necessary libraries using pip.
    ```python
    !pip install opendatasets pandas
    ```

2. **Data Loading**: Download and load the EEG Brainwave dataset.
    ```python
    import opendatasets as od
    import pandas as pd

    od.download("https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-feeling-emotions")
    data = pd.read_csv('/content/eeg-brainwave-dataset-feeling-emotions/emotions.csv')
    ```

3. **Preprocessing**:
    - Map emotion labels to numerical values.
    - Visualize the distribution of emotions using a pie chart.
    - Plot sample EEG time-series data and its power spectral density.
    - Generate a correlation heatmap for the features.

4. **Feature Extraction and Visualization**:
    - Use t-SNE for dimensionality reduction and visualization of high-dimensional EEG data.

5. **Capsule Network Implementation**:
    - Define the Capsule Network architecture using TensorFlow and Keras.
    - Compile the model with an appropriate optimizer and loss function.
    - Train the model on the EEG data, specifying the number of epochs and batch size.

6. **Evaluation**:
    - Predict the labels for the test set and calculate the accuracy.
    - Generate a confusion matrix and classification report to evaluate model performance.
    - Plot the confusion matrix for visual interpretation.
    - Perform random sampling to showcase the model’s predictions on individual test samples.

### Usage

To run the project locally:
1. Clone the repository.
2. Install the required libraries.
3. Download the EEG dataset from Kaggle.
4. Run the Jupyter notebook to preprocess data, train the model, and evaluate its performance.

