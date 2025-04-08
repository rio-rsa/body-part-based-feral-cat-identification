#------------------------------------------------------------------------------------------------------ Setup -------------------------------------------------------------------------------------------------------

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import pathlib

#--------------------- Mount to google drive.
from google.colab import drive

drive.mount('/content/gdrive')

#--------------------- Choose a single body part.
dataset_path = '/content/gdrive/MyDrive/dataset/Body'

data_dir = pathlib.Path(dataset_path)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------- Training & testing split ---------------------------------------------------------------------------------------------

#--------------------- Resizing
img_height,img_width=180,180
batch_size=32

#--------------------- Splitting data & creating labels
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.3,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_test_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.3,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

#--------------------- Further split the validation + test set into 50% validation, 50% test
val_batches = tf.data.experimental.cardinality(val_test_ds) // 2
val_ds = val_test_ds.take(val_batches)
test_ds = val_test_ds.skip(val_batches)

#--------------------- Verify the splits
print('Training batches:', tf.data.experimental.cardinality(train_ds).numpy())
print('Validation batches:', tf.data.experimental.cardinality(val_ds).numpy())
print('Test batches:', tf.data.experimental.cardinality(test_ds).numpy())

class_names = train_ds.class_names
print(class_names)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------- Training & testing split ---------------------------------------------------------------------------------------------

#--------------------- Model Setup & Train
resnet_model = Sequential()
#SETS UP A RESNET MODEL PRE TRAINED ON IMAGENET
pretrained_model= tf.keras.applications.ResNet50(include_top=False,
                   input_shape=(180,180,3),
                   pooling='avg',
                   weights='imagenet')
for layer in pretrained_model.layers:
        layer.trainable=False
  
#--------------------- Adds the necessary layers for a Base ResNet 50
resnet_model.add(pretrained_model)

# Add the final Dense layer for classification with 10 classes
resnet_model.add(Dense(10, activation='softmax'))

resnet_model.compile(optimizer=Adam(learning_rate=0.001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#--------------------- Training the model

epochs=10
history = resnet_model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

#--------------------- Evaluate the model on the test dataset
test_loss, test_accuracy = resnet_model.evaluate(test_ds)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

plt.show()

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
