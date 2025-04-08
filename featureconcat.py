#------------------------------------------------------------------------------------------------------ Setup -------------------------------------------------------------------------------------------------------

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten, Concatenate, Softmax, Input, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import pathlib

#--------------------- Mount to google drive.
from google.colab import drive
drive.mount('/content/gdrive')

#--------------------- Initialise the Data.
#--------------------- This is where the oombination of body parts is chosen in the Feature Concat Model.
#--------------------- You can add or remove initialisations to any of the body parts datasets here.
body_path = '/content/gdrive/MyDrive/dataset/Body'
front_leg_path = '/content/gdrive/MyDrive/dataset/Front Leg'
back_leg_path = '/content/gdrive/MyDrive/dataset/Back Leg'
tail_path = '/content/gdrive/MyDrive/dataset/Tail'

#--------------------- Set the directory variables.
body_dir = pathlib.Path(body_path)
front_leg_dir = pathlib.Path(front_leg_path)
back_leg_dir = pathlib.Path(back_leg_path)
tail_dir = pathlib.Path(tail_path)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------- Training, Testing & Validation Split ---------------------------------------------------------------------------------------

#--------------------- When adding a body part, you must split each body part data set.
#--------------------- If you remove a body part from above you must also remove the code for its dataset split here.
#----------- Body Data -----------
img_height,img_width=180,180
batch_size=32
#Splitting Data & Creating Labels
body_train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  body_dir,
  validation_split=0.3,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

body_val_test_ds = tf.keras.preprocessing.image_dataset_from_directory(
  body_dir,
  validation_split=0.3,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Further split the validation + test set into 50% validation, 50% test
body_val_batches = tf.data.experimental.cardinality(body_val_test_ds) // 2
body_val_ds = body_val_test_ds.take(body_val_batches)
body_test_ds = body_val_test_ds.skip(body_val_batches)

# Verify the splits
print('Training batches:', tf.data.experimental.cardinality(body_train_ds).numpy())
print('Validation batches:', tf.data.experimental.cardinality(body_val_ds).numpy())
print('Test batches:', tf.data.experimental.cardinality(body_test_ds).numpy())

#----------- Front Leg Data -----------
img_height,img_width=180,180
batch_size=32
#Splitting Data & Creating Labels
front_leg_train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  front_leg_dir,
  validation_split=0.3,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

front_leg_val_test_ds = tf.keras.preprocessing.image_dataset_from_directory(
  front_leg_dir,
  validation_split=0.3,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Further split the validation + test set into 50% validation, 50% test
front_leg_val_batches = tf.data.experimental.cardinality(front_leg_val_test_ds) // 2
front_leg_val_ds = front_leg_val_test_ds.take(front_leg_val_batches)
front_leg_test_ds = front_leg_val_test_ds.skip(front_leg_val_batches)

# Verify the splits
print('Training batches:', tf.data.experimental.cardinality(front_leg_train_ds).numpy())
print('Validation batches:', tf.data.experimental.cardinality(front_leg_val_ds).numpy())
print('Test batches:', tf.data.experimental.cardinality(front_leg_test_ds).numpy())

#----------- Back Leg Data -----------
img_height,img_width=180,180
batch_size=32
#Splitting Data & Creating Labels
back_leg_train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  back_leg_dir,
  validation_split=0.3,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

back_leg_val_test_ds = tf.keras.preprocessing.image_dataset_from_directory(
  back_leg_dir,
  validation_split=0.3,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

#--------------------- Further split the validation + test set into 50% validation, 50% test
back_leg_val_batches = tf.data.experimental.cardinality(back_leg_val_test_ds) // 2
back_leg_val_ds = back_leg_val_test_ds.take(back_leg_val_batches)
back_leg_test_ds = back_leg_val_test_ds.skip(back_leg_val_batches)

#--------------------- Verify the splits
print('Training batches:', tf.data.experimental.cardinality(back_leg_train_ds).numpy())
print('Validation batches:', tf.data.experimental.cardinality(back_leg_val_ds).numpy())
print('Test batches:', tf.data.experimental.cardinality(back_leg_test_ds).numpy())

#----------- Tail Data -----------
img_height,img_width=180,180
batch_size=32
#Splitting Data & Creating Labels
tail_train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  tail_dir,
  validation_split=0.3,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

tail_val_test_ds = tf.keras.preprocessing.image_dataset_from_directory(
  tail_dir,
  validation_split=0.3,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

#--------------------- Further split the validation + test set into 50% validation, 50% test
tail_val_batches = tf.data.experimental.cardinality(tail_val_test_ds) // 2
tail_val_ds = tail_val_test_ds.take(tail_val_batches)
tail_test_ds = tail_val_test_ds.skip(tail_val_batches)

#--------------------- Verify the splits
print('Training batches:', tf.data.experimental.cardinality(tail_train_ds).numpy())
print('Validation batches:', tf.data.experimental.cardinality(tail_val_ds).numpy())
print('Test batches:', tf.data.experimental.cardinality(tail_test_ds).numpy())

#--------------------- Get class names
front_leg_class_names = front_leg_train_ds.class_names
body_class_names = body_train_ds.class_names
back_leg_class_names = back_leg_train_ds.class_names
tail_class_names = tail_train_ds.class_names

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------ Extracting and saving features ------------------------------------------------------------------------------------------
#--------------------- Method to create the structure of the ResNet50 feature extractor
def create_base_model(name_prefix):
    base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(180, 180, 3), pooling='avg')

#--------------------- Create the model using the base model with pooling='avg'
    model = Model(inputs=base_model.input, outputs=base_model.output)

#--------------------- Renaming layers
    for layer in model.layers:
        layer._name = name_prefix + '_' + layer.name

    return model

#--------------------- Separate ResNet50 Feature Extractors
model_body = create_base_model('resnet_body')
model_front_leg = create_base_model('resnet_front_leg')
model_back_leg = create_base_model('resnet_back_leg')
model_tail = create_base_model('resnet_tail')

def extract_features(model, dataset, class_names):
    features = []
    labels = []
    image_batches = []
    for images, label in dataset:
        image_batches.append(images)  # Keep the images in case of visualization

#--------------------- Extracting features
        feature = model.predict(images)
        features.append(feature)
        labels.append(label)
    features = np.concatenate(features)
    labels = np.concatenate(labels)
    return features, labels, image_batches

#--------------------- If you remove a body part from above you must also remove the code here
body_train_features, body_train_labels, body_train_images = extract_features(model_body, body_train_ds, body_class_names)
body_val_features, body_val_labels, body_val_images = extract_features(model_body, body_val_ds, body_class_names)
body_test_features, body_test_labels, body_test_images = extract_features(model_body, body_test_ds, body_class_names)

front_leg_train_features, front_leg_train_labels, front_leg_train_images = extract_features(model_front_leg, front_leg_train_ds, front_leg_class_names)
front_leg_val_features, front_leg_val_labels, front_leg_val_images = extract_features(model_front_leg, front_leg_val_ds, front_leg_class_names)
front_leg_test_features, front_leg_test_labels, front_leg_test_images = extract_features(model_front_leg, front_leg_test_ds, front_leg_class_names)

back_leg_train_features, back_leg_train_labels, back_leg_train_images = extract_features(model_back_leg, back_leg_train_ds, back_leg_class_names)
back_leg_val_features, back_leg_val_labels, back_leg_val_images = extract_features(model_back_leg, back_leg_val_ds, back_leg_class_names)
back_leg_test_features, back_leg_test_labels, back_leg_test_images = extract_features(model_back_leg, back_leg_test_ds, back_leg_class_names)

tail_train_features, tail_train_labels, tail_train_images = extract_features(model_tail, tail_train_ds, tail_class_names)
tail_val_features, tail_val_labels, tail_val_images = extract_features(model_tail, tail_val_ds, tail_class_names)
tail_test_features, tail_test_labels, tail_test_images = extract_features(model_tail, tail_test_ds, tail_class_names)

#--------------------- Combining features from all body parts
train_features = np.concatenate([body_train_features, front_leg_train_features, back_leg_train_features, tail_train_features], axis=-1)
val_features = np.concatenate([body_val_features, front_leg_val_features, back_leg_val_features, tail_val_features], axis=-1)
test_features = np.concatenate([body_test_features, front_leg_test_features, back_leg_test_features, tail_test_features], axis=-1)

train_labels = body_train_labels
val_labels = body_val_labels
test_labels = body_test_labels

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------- Model training & testing ---------------------------------------------------------------------------------------------
input_shape = train_features.shape[1:]

#--------------------- Initialising the softmax classification layer
model = Sequential([
    Dense(10, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#--------------------- Training the softmax classification layer with the concatenated features
model.fit(train_features, train_labels, validation_data=(val_features, val_labels), epochs=10)

#--------------------- Evaluating the model
loss, accuracy = model.evaluate(test_features, test_labels)
print(f'Test accuracy: {accuracy}')
print(f'Test Loss: {loss}')

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

