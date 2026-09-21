
import tensorflow as tf
from tensorflow.keras import layers, models

def create(nclasses):
    model = models.Sequential([

        layers.Input(shape=(128, 128, 3)),

        #b1
        layers.Conv2D(
            filters = 32,
            kernel_size = (3, 3),
            padding = "same",
            activation = "relu",

        ),

        layers.MaxPooling2D(
            pool_size = (2,2)
        ),

        #b2
        layers.Conv2D(
            filters = 64,
            kernel_size = (3, 3),
            padding = "same",
            activation = "relu",
        ),

        layers.MaxPooling2D(
            pool_size = (2,2)
        ),

        #b3
        layers.Conv2D(
            filters = 128,
            kernel_size = (3, 3),
            padding = "same",
            activation = "relu",

        ),

        layers.MaxPooling2D(
            pool_size = (2,2)
        ),

        layers.Flatten(),

        layers.Dense(
            units = 100,
            activation = "relu"
        ),

        layers.Dropout(
            rate = 0.5,
        ),

        #o/p
        layers.Dense(
            nclasses,
            activation = "softmax"
        )
    ])

    return model