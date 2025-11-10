import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random

from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

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

hist_baseline = model_baseline.fit(x_train, y_train_cat,
                                   validation_data=(x_val, y_val_cat),
                                   epochs=5, batch_size=128)
