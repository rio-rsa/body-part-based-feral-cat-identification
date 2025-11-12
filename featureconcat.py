# SPDX-License-Identifier: MIT
# Copyright (C) 2025 Rio Rifqi Syah Akbar

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Flatten, Concatenate, Softmax, Input, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import pathlib
import gc
from collections import Counter

######################################################################################################################################################
# Setup
# Set a batch size here
batch_size = 32
# Set the directory variable for your cat images
folder_dir = Path('/content/gdrive/MyDrive/cat_images')
# Define the body parts you want to include
body_parts = ['back_leg', 'body', 'front_leg', 'tail']
datasets = {}

# declaring split folders
for part in body_parts:
    print(f"Loading datasets for {part}...")
    base_path = folder_dir/part
    datasets[part] = {
        'train_ds': tf.keras.preprocessing.image_dataset_from_directory(
            f'{base_path}/train',
            image_size=(180, 180),
            batch_size=batch_size,
            shuffle=False),
        'val_ds': tf.keras.preprocessing.image_dataset_from_directory(
            f'{base_path}/val',
            image_size=(180, 180),
            batch_size=batch_size,
            shuffle=False),
        'test_ds': tf.keras.preprocessing.image_dataset_from_directory(
            f'{base_path}/test',
            image_size=(180, 180),
            batch_size=batch_size,
            shuffle=False)
    }

# Get class names for active body parts
class_names = {}
for part_raw in body_parts:
    part_key = part_raw.replace('-', '_') # Use underscore for dictionary keys/model names
    class_names[part_key] = datasets[part_raw]['train_ds'].class_names

######################################################################################################################################################
# Feature Extraction
# ResNet50 feature extractor
def create_base_model(name_prefix):
    base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(180, 180, 3), pooling='avg')
    # Set the base model to be non-trainable
    base_model.trainable = False
    # Create the model using the base model with pooling='avg'
    model = Model(inputs=base_model.input, outputs=base_model.output)
    # Rename the layers to ensure uniqueness
    for layer in model.layers:
        layer._name = name_prefix + '_' + layer.name

    return model

# Create a separate model for each body part
models = {}
for part_raw in body_parts:
    part_key = part_raw.replace('-', '_')
    models[part_key] = create_base_model(f'resnet_{part_key}')

# Extract features
def extract_features(model, dataset, current_part_class_names):
    features = []
    labels = []
    for images, label in dataset:
        feature = model.predict(images, batch_size=batch_size)
        features.append(feature)
        labels.append(label)
    features = np.concatenate(features)
    labels = np.concatenate(labels)
    return features, labels

# Extract and save features
extracted_features_data = {part.replace('-', '_'): {} for part in body_parts}
all_train_features_list = []
all_val_features_list = []
all_test_features_list = []

# Take labels from 1 body part
primary_labels_part_raw = 'body'
if 'body' not in body_parts and len(body_parts) > 0:
    primary_labels_part_raw = body_parts[0]
primary_labels_part_key = primary_labels_part_raw.replace('-', '_')

train_labels = None
val_labels = None
test_labels = None

for part_raw in body_parts:
    part_key = part_raw.replace('-', '_')
    print(f"Extracting features for {part_key}...")

    extracted_features_data[part_key]['train_features'], extracted_features_data[part_key]['train_labels'], _ = \
        extract_features(models[part_key], datasets[part_raw]['train_ds'], class_names[part_key])
    extracted_features_data[part_key]['val_features'], extracted_features_data[part_key]['val_labels'], _ = \
        extract_features(models[part_key], datasets[part_raw]['val_ds'], class_names[part_key])
    extracted_features_data[part_key]['test_features'], extracted_features_data[part_key]['test_labels'], _ = \
        extract_features(models[part_key], datasets[part_raw]['test_ds'], class_names[part_key])

    all_train_features_list.append(extracted_features_data[part_key]['train_features'])
    all_val_features_list.append(extracted_features_data[part_key]['val_features'])
    all_test_features_list.append(extracted_features_data[part_key]['test_features'])

    if part_raw == primary_labels_part_raw:
        train_labels = extracted_features_data[part_key]['train_labels']
        val_labels = extracted_features_data[part_key]['val_labels']
        test_labels = extracted_features_data[part_key]['test_labels']

# Combine features from all active body parts
train_features = np.concatenate(all_train_features_list, axis=-1)
val_features = np.concatenate(all_val_features_list, axis=-1)
test_features = np.concatenate(all_test_features_list, axis=-1)

# Define the final model input shape based on combined features
input_shape = train_features.shape[1:]

######################################################################################################################################################
# Training & Testing

accuracy_results = []
loss_results = []

# Trains the model for 10 epochs then tests it. This code does it 5 times to find an average but can be deleted for just training it once.
for i in range(5):

  K.clear_session()
  gc.collect()

  model = Sequential([
      Dense(10, activation='softmax')
  ])

  model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

  # Train the model
  model.fit(train_features, train_labels, validation_data=(val_features, val_labels), epochs=10, batch_size=batch_size)

  # Evaluate the model
  loss, accuracy = model.evaluate(test_features, test_labels, batch_size=batch_size)
  print(f'Test accuracy: {accuracy} for iteration {i}')
  print(f'Test Loss: {loss} for iteration {i}')

  accuracy_results.append(accuracy)
  loss_results.append(loss)

  avg_accuracy = sum(accuracy_results) / len(accuracy_results)
  avg_loss = sum(loss_results) / len(loss_results)

  print(f'Average accuracy after {i} iterations: {avg_accuracy}')
  print(f'Average loss after {i} iterations: {avg_loss}')

######################################################################################################################################################
# Additional code to check how many images per body part per individual cat
# Loop through each active body part to print class counts dynamically

for part_raw in body_parts:
    part_key = part_raw.replace('-', '_')
    print(f"\n--- Class Counts for {part_raw} ---")

    # Train counts
    class_counts = Counter()
    for images, labels in datasets[part_raw]['train_ds']:
        labels = labels.numpy()
        for label in labels:
            class_counts[int(label)] += 1

    print(f"--- {part_raw} Train Set ---")
    for class_index, class_name in enumerate(class_names[part_key]):
        count = class_counts.get(class_index, 0)
        print(f"{class_name}: {count} train images")

    # Validation counts
    class_counts = Counter()
    for images, labels in datasets[part_raw]['val_ds']:
        labels = labels.numpy()
        for label in labels:
            class_counts[int(label)] += 1

    print(f"\n--- {part_raw} Validation Set ---")
    for class_index, class_name in enumerate(class_names[part_key]):
        count = class_counts.get(class_index, 0)
        print(f"{class_name}: {count} val images")

    # Test counts
    class_counts = Counter()
    for images, labels in datasets[part_raw]['test_ds']:
        labels = labels.numpy()
        for label in labels:
            class_counts[int(label)] += 1

    print(f"\n--- {part_raw} Test Set ---")
    for class_index, class_name in enumerate(class_names[part_key]):
        count = class_counts.get(class_index, 0)
        print(f"{class_name}: {count} test images")

