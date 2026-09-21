from models.cnn import create
import tensorflow as tf
from data_loader.gdata import train_dataset, val_dataset, test_dataset

nclasses = 43

model = create(nclasses)

model.compile(
    optimizer = "adam",
    loss = tf.keras.losses.sparse_categorical_crossentropy,
    metrics = ["accuracy"]
)

model.summary()

#traininggg
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10
)

loss, accuracy = model.evaluate(test_dataset)

print("Test loss", loss)
print("Test accuracy", accuracy)

model.save("german_cnn.keras")