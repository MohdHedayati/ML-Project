from models.cnn import create
import tensorflow as tf
from data_loader.bdata import train_dataset, val_dataset, test_dataset

model = create(62)

model.compile(
    optimizer = "adam",
    loss = tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics = ["accuracy"]
)

model.summary()

history = model.fit(
    train_dataset,
    validation_data = val_dataset,
    epochs = 10
)

loss, acc = model.evaluate(test_dataset)
print("Test Loss: ", loss)
print("Test Accuracy: ", acc)


model.save("belgium_cnn.keras")