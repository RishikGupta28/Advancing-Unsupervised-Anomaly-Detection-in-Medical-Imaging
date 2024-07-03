### MedAI AnomalyPro: Advancing Unsupervised Anomaly Detection in Medical Imaging

This repository contains the implementation of MedAI AnomalyPro, a novel approach to unsupervised anomaly detection in medical imaging. The method leverages a unidirectional image-to-image translation technique rooted in Generative Adversarial Networks (GANs) to detect anomalies in various medical imaging modalities, including MRIs and X-rays.

### Project Outline

1. **Introduction**
2. **Literature Review**
3. **MedAI AnomalyPro: Proposed Method**
    - Framework
    - Discriminator Objective
    - Generator Objective
4. **Experiments and Results**
    - Covid-19 Recognition
    - Chest X-ray 14: Identification of Diseases
    - Migraine Detection
5. **Conclusion**

### Dataset Details

- **COVID-x Dataset**: Contains 200 Chest X-rays in the testing set (100 positives, 100 healthy) and 15,464 in the training set (1,670 COVID-19 positives, 13,794 healthy).
- **ChestX-ray14 Dataset**: Focuses on Posterior Anterior (PA) images associated with 14 thoracic disorders.
- **Migraine Dataset**: Includes 104 brain MRIs of healthy individuals and 96 brain MRIs of migraine sufferers.

### Tech Stack

- **Programming Language**: Python
- **Libraries and Tools**:
  - `tensorflow` and `keras`: For building and training the GAN model.
  - `pandas`: For data manipulation and analysis.
  - `numpy`: For numerical computations.
  - `matplotlib` and `seaborn`: For data visualization.

### Algorithm and Approach

1. **Framework**: MedAI AnomalyPro consists of a discriminator and a generator network.
    - **Discriminator**: Aims to distinguish between authentic medical images and synthetically manufactured anomaly-free images.
    - **Generator**: Transforms input medical images into anomaly-free representations using a composite dataset.

2. **Discriminator Objective**:
    - **Adversarial Loss**: Measures the discriminator's ability to differentiate between real and generated images.
    - **Gradient Penalty Term**: Enhances training stability by penalizing the gradients of the discriminator's output.

3. **Generator Objective**:
    - **Adversarial Loss**: Measures the generator's ability to produce images classified as genuine by the discriminator.
    - **Identity Loss**: Ensures the generator preserves the identity of healthy input images.
    - **Reconstruction Loss**: Encourages the generator to produce outputs close to the input images.
    - **Focus Loss**: Controls the alignment of mask values to ensure they are closer to 0 or 1.

### Usage

To run the project locally:
1. Clone the repository.
2. Install the required libraries.
3. Download the datasets (COVID-x, ChestX-ray14, Migraine).
4. Run the provided scripts to preprocess data, train the model, and evaluate its performance.

### Example Code Snippets

#### Environment Setup
```python
!pip install tensorflow pandas matplotlib seaborn
```

#### Data Loading
```python
import pandas as pd

# Load datasets
covid_data = pd.read_csv('path/to/covid-dataset.csv')
chestxray_data = pd.read_csv('path/to/chestxray14-dataset.csv')
migraine_data = pd.read_csv('path/to/migraine-dataset.csv')
```

#### Framework Definition
```python
from tensorflow.keras import layers, models

def build_discriminator():
    input_img = layers.Input(shape=(image_shape))
    # Define the architecture
    model = models.Model(inputs=input_img, outputs=output)
    return model

def build_generator():
    input_img = layers.Input(shape=(image_shape))
    # Define the architecture
    model = models.Model(inputs=input_img, outputs=output)
    return model

discriminator = build_discriminator()
generator = build_generator()
```

#### Training the Model
```python
epochs = 50
batch_size = 32

for epoch in range(epochs):
    for batch in range(0, len(train_data), batch_size):
        # Train discriminator and generator
        pass
```

#### Evaluation
```python
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

predictions = generator.predict(test_data)
auc = roc_auc_score(test_labels, predictions)
precision = precision_score(test_labels, predictions)
recall = recall_score(test_labels, predictions)
f1 = f1_score(test_labels, predictions)

print(f'AUC: {auc}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}')
```

### Results

- **Covid-19 Recognition**: Achieved an AUC of 0.84, with precision, specificity, and F1 scores of 0.76.
- **Chest X-ray 14**: Achieved the highest detection AUC score of 0.56, outperforming f-AnoGAN.
- **Migraine Detection**: Achieved an AUC value of 0.75, with recall, accuracy, specificity, and F1 scores of 0.79, 0.71, 0.81, and 0.74, respectively.

