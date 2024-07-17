# Mini Project Description: Hot Dog or Not Hot Dog Classifier Using CNN
## Objective:
To create a convolutional neural network (CNN) that can classify images as "hot dog" or "not hot dog." The project involves data preprocessing, model building, training, and evaluation.

## Project Overview:
### Data Collection and Preparation:

Collect a dataset of images containing hot dogs and not hot dogs.
Split the dataset into training, validation, and test sets.
Apply data augmentation to enhance the training dataset's diversity.

### Building the CNN Model:

Construct a CNN model using TensorFlow and Keras.
Include layers for convolution, pooling, dropout, and dense layers to create an effective architecture.

### Training the Model:

Compile the model with appropriate optimizer, loss function, and metrics.
Train the model using the training dataset and validate it using the validation dataset.
Monitor the training process and make necessary adjustments to avoid overfitting.

### Evaluating the Model:

Evaluate the model on the test dataset to measure its accuracy and other metrics.
Visualize the performance of the model using plots.


## Detailed Steps:

### Data Collection and Preparation:

#### Dataset: 
You can use the "Food-101" dataset or any other image dataset with hot dogs and non-hot dog images.
#### Preprocessing:
Resize images to a consistent size.
Normalize pixel values to the range [0, 1].
Split the dataset into training (80%), validation (10%), and test (10%) sets.
Apply data augmentation techniques such as random flips and rotations.
#### Building the CNN Model:
```python
import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])
````
### Training the Model:
```python
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

history = model.fit(train_ds,
                    validation_data=valid_ds,
                    epochs=50,
                    verbose=1)
````
### Evaluating the Model:
```python
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test Accuracy: {test_acc}")

 ```
### Visualize Training and Validation Accuracy and Loss
```python
plt.figure(figsize=(10, 10))
for image_batch, label_batch in valid_ds.take(1): #takes one batch of data from the validation dataset.
  images = image_batch
  labels = label_batch
````
## Summary
This mini project involves creating a CNN to classify images as "hot dog" or "not hot dog," starting from data collection and preprocessing, building and training the model, to evaluating and deploying the model. It covers key steps in a machine learning project and provides a practical application of convolutional neural networks.






