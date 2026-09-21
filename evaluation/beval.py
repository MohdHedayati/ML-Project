import tensorflow as tf
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report

from data_loader.bdata import test_dataset

model = tf.keras.models.load_model("belgium_cnn.keras")

actualY = []
for img, lbl in test_dataset:
    actualY.extend(lbl.numpy())

actualY = tf.convert_to_tensor(actualY)

predY = model.predict(test_dataset)

predY = predY.argmax(axis = 1)


#Metrics
accuracy = accuracy_score(actualY, predY)
precision = precision_score(actualY, predY, average="weighted", zero_division = 0)
recall = recall_score(actualY, predY, average="weighted", zero_division = 0)
f1 = f1_score(actualY, predY, average="weighted", zero_division = 0)
f1Macro = f1_score(actualY, predY, average="macro")

print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1)
print("F1 Macro: ", f1Macro)

import matplotlib.pyplot as plt
import seaborn as sn

confuse = confusion_matrix(actualY, predY)

plt.figure(figsize=(20, 20))

sn.heatmap(
    confuse,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()

print("\nClassification Report:")
print(classification_report(actualY, predY, zero_division = 0))