import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import seaborn as sns
import pandas as pd

from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


#Loading the data
(x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
print("raw shapes:", x_train_full.shape, y_train_full.shape, x_test.shape, y_test.shape)
print("dtype:", x_train_full.dtype, "min/max:", x_train_full.min(), x_train_full.max())

#Convert to float and normalize
x_train_full = x_train_full.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

#Flatten the y arrays (Remove a dimension)
y_train_full = y_train_full.flatten()
y_test = y_test.flatten()

# Split sample images; 5000 for validation, 45000 for training
# Stratify to provide an even sample of each image class 
x_train, x_val, y_train, y_val = train_test_split(
    x_train_full, y_train_full, test_size=5000, random_state=42, stratify=y_train_full
)

# Apply one-hot encoding to class labels
y_train_cat = to_categorical(y_train, num_classes=10)
y_val_cat = to_categorical(y_val, num_classes=10)
y_test_cat = to_categorical(y_test, num_classes=10)

print("train/val/test shapes:", x_train.shape, x_val.shape, x_test.shape)

# 3. Show a few images from training set that the model will be training off of
class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
plt.figure(figsize=(8,4))
for i in range(10):
    idx = random.randint(0, len(x_train)-1)
    plt.subplot(2,5,i+1)
    plt.imshow(x_train[idx])
    plt.title(class_names[y_train[idx]])
    plt.axis('off')
plt.tight_layout()
plt.show()

print("About to build baseline model...")

# 4. Basic Neural Network Model
flat_input_shape = (32*32*3,)
model_baseline = models.Sequential([
    layers.Input(shape=(32,32,3)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
model_baseline.compile(optimizer=optimizers.Adam(1e-3),
                       loss='categorical_crossentropy', metrics=['accuracy'])
model_baseline.summary()

print("Baseline model built; about to train...")


hist_baseline = model_baseline.fit(x_train, y_train_cat,
                                   validation_data=(x_val, y_val_cat),
                                   epochs=5, batch_size=128)

print("Baseline model training finished!")

print("About to build CNN...")

# 5. build a small CNN
def build_cnn():
    model = models.Sequential()
    # conv block 1: learns low-level filters like edges
    model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(32,32,3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.25))

    # conv block 2: learns more abstract features
    model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.25))

    # classification head
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))
    return model

cnn = build_cnn()
cnn.compile(optimizer=optimizers.Adam(1e-3),
            loss='categorical_crossentropy', metrics=['accuracy'])
cnn.summary()

#6 Data augmentation
print("Starting CNN augmented training...")

datagen = ImageDataGenerator(
    rotation_range=12,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)
datagen.fit(x_train)

# use generator to train
history_aug = cnn.fit(datagen.flow(x_train, y_train_cat, batch_size=64),
                      validation_data=(x_val, y_val_cat),
                      steps_per_epoch=len(x_train)//64,
                      epochs=30)

#7 resize images for transfer learning
IMAGE_SIZE = (96,96)
x_train_t = tf.image.resize(x_train, IMAGE_SIZE).numpy()
x_val_t = tf.image.resize(x_val, IMAGE_SIZE).numpy()
x_test_t = tf.image.resize(x_test, IMAGE_SIZE).numpy()

# build TL model
base = tf.keras.applications.MobileNetV2(input_shape=IMAGE_SIZE+(3,), include_top=False, weights='imagenet', pooling='avg')
base.trainable = False  # freeze base

inputs = layers.Input(shape=IMAGE_SIZE+(3,))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)  # matches training preprocessing
x = base(x, training=False)
x = layers.Dropout(0.4)(x)
x = layers.Dense(128, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax')(x)
tl_model = models.Model(inputs, outputs)

tl_model.compile(optimizer=optimizers.Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
tl_model.summary()

# train only head
history_tl = tl_model.fit(x_train_t, y_train_cat, validation_data=(x_val_t, y_val_cat), epochs=8, batch_size=64)



#8 predict with CNN (which expects 32x32)
y_pred_proba = cnn.predict(x_test)
y_pred = np.argmax(y_pred_proba, axis=1)
print("CNN test accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=class_names))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('predicted'); plt.ylabel('true'); plt.title('CNN Confusion Matrix')
plt.show()

#9 Save models
os.makedirs("models", exist_ok=True)
cnn.save("models/cnn_from_scratch.h5")
tl_model.save("models/mobilenetv2_head.h5")

# save history to CSV
pd.DataFrame(history_aug.history).to_csv("cnn_history.csv", index=False)
pd.DataFrame(history_tl.history).to_csv("tl_history.csv", index=False)
