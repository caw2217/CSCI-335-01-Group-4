import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib

from skimage.feature import hog
from skimage import data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from tensorflow.keras.datasets import cifar10
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# === Load CIFAR-10 ===
# X: uint8 images of shape (32, 32, 3); y: integer labels 0-9
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = y_train.flatten()
y_test = y_test.flatten()

class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = tf.image.rgb_to_grayscale(x_train)
x_test = tf.image.rgb_to_grayscale(x_test)

print("Train:", x_train.shape, y_train.shape)
print("Test :", x_test.shape, y_test.shape)
print(y_train.shape[0])

# === Visualize sample images from CIFAR-10 ===
# Display 25 random training images
plt.figure(figsize=(6,6))
for i in range(25):
    idx = np.random.randint(0, len(x_train))
    plt.subplot(5,5,i+1)
    plt.imshow(x_train[idx], cmap='gray')
    plt.title(class_names[y_train[idx]])
    plt.axis("off")
plt.suptitle("CIFAR-10 after grayscale", fontsize=14)
plt.tight_layout()
plt.show()

# === Utilities: training curves and evaluation ===

def plot_training_curves(history, title_prefix="Model"):
    """Plot training & validation accuracy and loss (separate charts)."""
    # Accuracy plot
    plt.figure()
    plt.plot(history.history.get("accuracy", []), label="train_acc")
    if "val_accuracy" in history.history:
        plt.plot(history.history["val_accuracy"], label="val_acc")
    plt.title(f"{title_prefix} — Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    # Loss plot
    plt.figure()
    plt.plot(history.history.get("loss", []), label="train_loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="val_loss")
    plt.title(f"{title_prefix} — Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def evaluate_model(model, x_test, y_test, class_names=None, title="Evaluation"):
    """Common evaluation function: prints loss/accuracy and a small sample of predictions."""
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"{title} — Test Loss: {loss:.4f} | Test Accuracy: {acc:.4f}")

    # Show a few predictions
    idx = np.random.choice(len(x_test), size=8, replace=False)
    x_sample, y_true = x_test[idx], y_test[idx]
    y_prob = model.predict(x_sample, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    return loss, acc

## Feature Extraction

features = []

for img in x_train:
    img_arr = np.array(img).squeeze()
    features.append(hog(img_arr))

features_test = []

for img in x_test:
    img_arr = np.array(img).squeeze()
    features_test.append(hog(img_arr))

x_train_h = np.array(features)
x_test_h = np.array(features_test)

print(x_train_h.shape)
print(x_test_h.shape)

##Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(x_train_h, y_train)

y_pred = rf.predict(x_test_h)

accuracy = rf.score(x_test_h, y_test)
print("Accuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

print("Classification Report:", classification_report(y_test, y_pred))

##SVM

svm = LinearSVC(C=1, max_iter=5000)
svm.fit(x_train_h, y_train)

y_pred = svm.predict(x_test_h)

accuracy = svm.score(x_test_h, y_test)
print("Accuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

print("Classification Report:", classification_report(y_test, y_pred))
