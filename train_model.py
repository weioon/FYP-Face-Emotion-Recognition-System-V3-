import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import tensorflow as tf

# Ensure TensorFlow uses the GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU is available and will be used for training.")
else:
    print("GPU is not available. Training will use the CPU.")

# -------------------------------
# 1. Set Paths and Parameters
# -------------------------------
# Use raw strings (prefix with r) for Windows paths.
train_dir = r"C:\Users\weioo\Downloads\Facial_Emotion_Dataset-main\Facial_Emotion_Dataset-main\train_dir"
valid_dir = r"C:\Users\weioo\Downloads\Facial_Emotion_Dataset-main\Facial_Emotion_Dataset-main\test_dir"

img_height, img_width = 224, 224
batch_size = 32

# -------------------------------
# 2. Data Augmentation
# -------------------------------
# For training, we apply several augmentations to boost dataset diversity.
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# For validation, only rescale.
valid_datagen = ImageDataGenerator(rescale=1./255)

# Create generators that read images from the directories.
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# -------------------------------
# 3. Model Definition and Initial Training
# -------------------------------
# Use VGG16 as the base model, excluding its top layers.
base_model = VGG16(weights='imagenet', include_top=False,
                   input_shape=(img_height, img_width, 3))

# Add a new head for emotion classification.
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Initially freeze all layers of the base model.
for layer in base_model.layers:
    layer.trainable = False

# Compile the model using Adam optimizer.
model.compile(optimizer=Adam(lr=1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Set up callbacks to help during training.
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
]

# Train the new head for a number of epochs.
initial_epochs = 20
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=valid_generator,
    validation_steps=valid_generator.samples // batch_size,
    epochs=initial_epochs,
    callbacks=callbacks
)

# -------------------------------
# 4. Fine-Tuning the Model
# -------------------------------
# Unfreeze the last few layers of the base model for fine-tuning.
# (Adjust the slice as needed; here we unfreeze the last 4 layers.)
for layer in base_model.layers[-4:]:
    layer.trainable = True

# Re-compile the model with a lower learning rate for fine-tuning.
model.compile(optimizer=SGD(lr=1e-4, momentum=0.9),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Continue training (fine-tuning) for additional epochs.
fine_tune_epochs = 10
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=valid_generator,
    validation_steps=valid_generator.samples // batch_size,
    epochs=fine_tune_epochs,
    callbacks=callbacks
)

# -------------------------------
# 5. Save the Final Model
# -------------------------------
# Save your updated model; your inference scripts (app.py and realtime_emotion.py)
# can then load this file.
model.save('Final_model.h5')