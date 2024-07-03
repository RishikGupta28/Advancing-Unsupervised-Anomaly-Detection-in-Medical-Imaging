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

### Example Code Snippets

#### Loading Data
```python
import opendatasets as od
import pandas as pd

od.download("https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-feeling-emotions")
data = pd.read_csv('/content/eeg-brainwave-dataset-feeling-emotions/emotions.csv')
```

#### Preprocessing
```python
label_mapping = {'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2}
data['label'] = data['label'].map(label_mapping)
```

#### Visualizing Data Distribution
```python
emotion_counts = data['label'].value_counts()
emotion_labels = {0: 'NEGATIVE', 1: 'NEUTRAL', 2: 'POSITIVE'}
labels = [emotion_labels[label] for label in emotion_counts.index]

plt.figure(figsize=(8, 8))
plt.pie(emotion_counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=['red', 'yellow', 'green'])
plt.title("Distribution of Emotions (0: NEGATIVE, 1: NEUTRAL, 2: POSITIVE)")
plt.axis('equal')
plt.show()
```

#### Capsule Network Definition
```python
from tensorflow.keras import layers, models

input_layer = layers.Input(shape=(750,))
primary_caps = layers.Dense(128, activation='relu')(input_layer)
capsule_layer = layers.Dense(32, activation='relu')(primary_caps)
output_layer = layers.Dense(3, activation='softmax')(capsule_layer)

capsule_net = models.Model(inputs=input_layer, outputs=output_layer)
capsule_net.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

#### Training the Model
```python
epochs = 10
batch_size = 32

capsule_net.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size)
```

#### Model Evaluation
```python
from sklearn.metrics import confusion_matrix, classification_report

predictions = capsule_net.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1)

cm = confusion_matrix(y_test, predicted_labels)
clr = classification_report(y_test, predicted_labels, target_names=label_mapping.keys())

plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=label_mapping.keys(), yticklabels=label_mapping.keys())
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

print("Classification Report:\n", clr)
```
