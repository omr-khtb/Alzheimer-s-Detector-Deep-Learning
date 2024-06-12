import os
import zipfile
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator as IDG
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
import glob
import io
import sys

# Disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Set UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Define the local path to the dataset
dataset_path = 'archive.zip'
extract_path = 'datasets/alzheimers-dataset/'

# Expand the user directory symbol '~'
dataset_path = os.path.expanduser(dataset_path)
extract_path = os.path.expanduser(extract_path)

# Unzip the dataset if it hasn't been unzipped already
if not os.path.exists(extract_path):
    os.makedirs(extract_path)
    
with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Define the paths for training and testing directories
extracted_dir = os.path.join(extract_path, 'Alzheimer_s Dataset')
train_dir = os.path.join(extracted_dir, 'train')
test_dir = os.path.join(extracted_dir, 'test')

# Print TensorFlow version
print("TensorFlow Version:", tf.__version__)

# Set constants and parameters
CLASSES = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
IMG_SIZE = 224
IMAGE_SIZE = [224, 224]
DIM = (IMG_SIZE, IMG_SIZE)
BATCH_SIZE = 32

# Set a random seed for reproducibility
seed_value = 1337

# Create separate data generators for training and validation with augmentation
train_datagen = IDG(
    rescale=1./255,
    zoom_range=[.99, 1.01],
    horizontal_flip=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    fill_mode='constant',
    validation_split=0.2,
    data_format='channels_last'
)

train_ds = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',  # Assuming you're doing multi-class classification
    subset='training',
    seed=seed_value
)

val_ds = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',  # Assuming you're doing multi-class classification
    subset='validation',
    seed=seed_value
)

# Count images per category
datasets = ["train", "test"]
counts = {cls: 0 for cls in CLASSES}
for dataset in datasets:
    for cls in CLASSES:
        counts[cls] += len(glob.glob(os.path.join(extracted_dir, dataset, cls, '*.jpg')))

data = {'Category': list(counts.keys()), 'Count': list(counts.values())}
sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
ax = sns.barplot(x='Category', y='Count', data=data, palette='viridis')
ax.set_title('Number of Images per Category')
plt.show()

# Class weights
total_samples = sum(counts.values())
class_weights = {i: total_samples / (4 * count) for i, count in enumerate(counts.values())}

# New section: MobileNet model with the same data augmentation
# Load the MobileNet model
base_model = MobileNet(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")

# Freeze the base model
for layer in base_model.layers:
    layer.trainable = False

# Create a new model on top
model_mobilenet = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    BatchNormalization(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    BatchNormalization(),
    Dense(len(CLASSES), activation='softmax')
])

# Compile the model
model_mobilenet.compile(optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])

# Print the model summary
model_mobilenet.summary()

# Set up callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

# Train the model
history_mobilenet = model_mobilenet.fit(
    train_ds,
    steps_per_epoch=train_ds.samples // BATCH_SIZE,
    epochs=50,
    validation_data=val_ds,
    validation_steps=val_ds.samples // BATCH_SIZE,
    class_weight=class_weights,  # Utilize the calculated class weights
    callbacks=[reduce_lr]
)

# Save the model
model_save_path = 'datasets/alzheimers-dataset/mobilenet_model_our.h5'
model_save_path = os.path.expanduser(model_save_path)
model_mobilenet.save(model_save_path)
print(f"Model saved to {model_save_path}")

# Evaluate the model on the test set
test_datagen = IDG(rescale=1./255)

test_ds = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    seed=seed_value
)

test_results_mobilenet = model_mobilenet.evaluate(test_ds)
print("Test Loss:", test_results_mobilenet[0])
print("Test Accuracy:", test_results_mobilenet[1])

# Plot training history
fig, ax = plt.subplots(1, 2, figsize=(20, 5))
ax = ax.ravel()

for i, met in enumerate(['accuracy', 'loss']):
    ax[i].plot(history_mobilenet.history[met])
    ax[i].plot(history_mobilenet.history['val_' + met])
    ax[i].set_title(f'MobileNet Model {met.capitalize()}')
    ax[i].set_xlabel('Epochs')
    ax[i].set_ylabel(met.capitalize())
    ax[i].legend(['Train', 'Validation'])

plt.show()
