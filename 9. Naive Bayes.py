import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

faces = fetch_olivetti_faces()
X = faces.data
y = faces.target
images = faces.images

X_train, X_test, y_train, y_test, img_train, img_test = train_test_split(X, y, images, test_size=0.3, random_state=42, stratify=y)
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy)
def show_predictions(images, true_labels, predicted_labels, n=8):
    plt.figure(figsize=(15, 5))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(f"True: {true_labels[i]}\nPred: {predicted_labels[i]}")
        plt.axis('off')
    plt.suptitle("Sample Test Predictions")
    plt.show()
show_predictions(img_test, y_test, y_pred, n=8)