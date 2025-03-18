import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Activation, Multiply
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import os
import time
from keras import backend as K

# Configure GPU memory growth to avoid memory issues
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print("GPU available, configuring TensorFlow...")
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print(f"Using GPU: {physical_devices}")
else:
    print("No GPU found, using CPU. This will be slow for training deep learning models.")

# Parameters
img_size = 48
batch_size = 64  # Increase if you have enough GPU memory
epochs = 50
learning_rate = 0.001
data_dir = r'C:\Users\weioo\OneDrive - UNIVERSITY UTARA MALAYSIA\Desktop\FYP-Face-Emotion-Recognition-System-V3-\FER dataset\train'
# -------------------------------
# 1. Enhanced Data Augmentation
# -------------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2],
    validation_split=0.2
)

valid_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Load data
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_size, img_size),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

valid_generator = valid_datagen.flow_from_directory(
    data_dir,
    target_size=(img_size, img_size),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# After line 74, add class weights to handle imbalanced classes
# This helps improve accuracy for underrepresented emotions like "Disgust"
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Calculate class weights
classes = list(train_generator.class_indices.keys())
class_counts = [0] * len(classes)
for i in range(len(train_generator.classes)):
    class_counts[train_generator.classes[i]] += 1

class_weights = {}
total_samples = sum(class_counts)
for i, count in enumerate(class_counts):
    # Inversely proportional to class frequency
    class_weights[i] = total_samples / (len(class_counts) * count)
print("Class weights:", class_weights)

# In train_model.py, add targeted augmentation
# After calculating class weights:

# Identify classes with fewer samples
min_count = min(class_counts)
min_class_idx = class_counts.index(min_count)
min_class_name = classes[min_class_idx]
print(f"Underrepresented class: {min_class_name} with {min_count} samples")

# For the disgust class specifically, create more augmented samples
if 'disgust' in [c.lower() for c in classes]:
    disgust_idx = [i for i, c in enumerate(classes) if c.lower() == 'disgust'][0]
    class_weights[disgust_idx] *= 1.5  # Extra weight for disgust class

# -------------------------------
# 2. Build Model with MobileNetV2 + Modifications
# -------------------------------
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(img_size, img_size, 3)
)

# Convert grayscale to RGB (MobileNetV2 expects 3 channels)
def grayscale_to_rgb(x):
    return tf.repeat(x, 3, axis=-1)

# Create the model architecture
inputs = tf.keras.Input(shape=(img_size, img_size, 1))
x = tf.keras.layers.Lambda(grayscale_to_rgb)(inputs)
x = base_model(x)
x = GlobalAveragePooling2D()(x)

# Get the output dimension from the previous layer
features_dim = K.int_shape(x)[-1]  # This will be 1280

# Create attention with matching dimensions
attention = Dense(features_dim, activation='relu')(x)  # Change 512 to features_dim
attention = Dense(features_dim, activation='sigmoid')(attention)  # Change 512 to features_dim
x = Multiply()([x, attention])

# Add batch normalization
x = Dense(512)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)

# Try smaller dropout for the second dense layer
x = Dense(256)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.3)(x)

# Use label smoothing for better generalization
outputs = Dense(7, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile model with mixed precision for faster GPU training
tf.keras.mixed_precision.set_global_policy('mixed_float16')
model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

# -------------------------------
# 3. Enhanced Callbacks
# -------------------------------
# Add timing callback to measure performance
class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
        self.epoch_time_start = time.time()
        print(f"Epoch took {self.times[-1]:.2f} seconds")

time_callback = TimeHistory()

# Rest of the callbacks
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1),
    time_callback
]

# -------------------------------
# 4. Initial Training
# -------------------------------
start_time = time.time()
history_initial = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=valid_generator,
    validation_steps=valid_generator.samples // batch_size,
    epochs=25,
    callbacks=callbacks,
    class_weight=class_weights  # Add this line
)

# -------------------------------
# 5. Fine-tuning
# -------------------------------
# Unfreeze some layers for fine-tuning
for layer in base_model.layers[-15:]:  # Unfreeze more layers for better fine-tuning
    layer.trainable = True

# Lower learning rate for fine-tuning
model.compile(
    optimizer=SGD(learning_rate=1e-5, momentum=0.9),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Continue training
history_fine_tune = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=valid_generator,
    validation_steps=valid_generator.samples // batch_size,
    epochs=25,
    callbacks=callbacks,
    class_weight=class_weights  # Add this line
)

total_time = time.time() - start_time
print(f"Total training time: {total_time/60:.2f} minutes")

# -------------------------------
# 6. Save Final Model
# -------------------------------
model.save('Backend/Final_model.h5')  # Save to Backend directory
print("Model saved to Backend/Final_model.h5")

# -------------------------------
# 7. Plot training results
# -------------------------------
def plot_history(history_initial, history_fine_tune=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history_initial.history['accuracy'])
    ax1.plot(history_initial.history['val_accuracy'])
    if history_fine_tune:
        offset = len(history_initial.history['accuracy'])
        ax1.plot(range(offset, offset + len(history_fine_tune.history['accuracy'])), 
                history_fine_tune.history['accuracy'])
        ax1.plot(range(offset, offset + len(history_fine_tune.history['val_accuracy'])), 
                history_fine_tune.history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation', 'Train (fine-tune)', 'Validation (fine-tune)'], loc='lower right')
    
    # Plot loss
    ax2.plot(history_initial.history['loss'])
    ax2.plot(history_initial.history['val_loss'])
    if history_fine_tune:
        offset = len(history_initial.history['loss'])
        ax2.plot(range(offset, offset + len(history_fine_tune.history['loss'])), 
                history_fine_tune.history['loss'])
        ax2.plot(range(offset, offset + len(history_fine_tune.history['val_loss'])), 
                history_fine_tune.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation', 'Train (fine-tune)', 'Validation (fine-tune)'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

plot_history(history_initial, history_fine_tune)

# Then update all scripts to use absolute paths:
import os
model_path = os.path.join(os.path.dirname(__file__), 'Final_model.h5')
emotion_model = load_model(model_path)