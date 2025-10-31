

# ------------------------------
# MNIST Classification (Sklearn)
# ------------------------------

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score
)

# ------------------------------
# 1. Load MNIST dataset from OpenML
# ------------------------------
print(" Downloading MNIST dataset from OpenML...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

x, y = mnist['data'], mnist['target'].astype(int)
x = x / 255.0  # normalize

# Split data (first 60k for train, last 10k for test)
x_train, x_test = x[:60000], x[60000:]
y_train, y_test = y[:60000], y[60000:]

print(f" Data loaded: Train={x_train.shape}, Test={x_test.shape}")

# ------------------------------
# 2. Train Logistic Regression model
# ------------------------------
print("\n Training Logistic Regression model...")
model = LogisticRegression(max_iter=100, solver='lbfgs', n_jobs=-1)
model.fit(x_train, y_train)

# ------------------------------
# 3. Evaluate the model
# ------------------------------
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n Test Accuracy: {acc:.4f}")

# ------------------------------
# 4. Classification metrics
# ------------------------------
report = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()

print("\n Precision, Recall, F1-score, and Support per class:")
print(df_report.round(3))

# ------------------------------
# 5. Confusion matrix
# ------------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# ------------------------------
# 6. Visualize first 5 test images with predictions
# ------------------------------
plt.figure(figsize=(10, 2))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"True: {y_test[i]}\nPred: {y_pred[i]}")
    plt.axis('off')
plt.suptitle("First 5 Test Samples and Predictions", fontsize=14)
plt.show()

# ------------------------------
# 7. Misclassified samples
# ------------------------------
misclassified_idx = np.where(y_test != y_pred)[0]
print(f"\n Total Misclassified Samples: {len(misclassified_idx)}")

plt.figure(figsize=(10, 5))
for i, idx in enumerate(misclassified_idx[:10]):  # show first 10 misclassified
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
    plt.title(f"T:{y_test[idx]} | P:{y_pred[idx]}")
    plt.axis('off')
plt.suptitle("Example Misclassified Samples", fontsize=14)
plt.show()
