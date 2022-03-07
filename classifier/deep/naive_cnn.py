# https://keras.io/examples/vision/image_classification_from_scratch/
# Xception Model
from tensorflow import keras
import os
from keras import layers
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from collections import defaultdict
import random
import argparse


config = {
    "batch_size": 32,
    "epoch": 50,
    "lr": 1e-3,
}


class BasicLoader(object):
    def __init__(self, root, batch_size):
        self.root = root
        self.batch_size = batch_size
        self.start_id = 0
        self.label2data, self.label2id, self.id2label, self.num_class, self.data_array = self.load()
        print("Data Loaded.")
        self.num_class = len(self.label2id)

    def load(self):
        label2data = defaultdict(list)
        for d in os.listdir(self.root):
            sub_dir = os.path.join(self.root, d)
            if os.path.isfile(sub_dir):
                continue
            for file in os.listdir(sub_dir):
                if file.endswith(".jpg"):
                    path = os.path.join(sub_dir, file)
                    image = plt.imread(path)
                    if len(image.shape) == 2:
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                    label2data[d].append(image)
        label2id = {label: i for i, label in enumerate(label2data.keys())}
        id2label = {i: label for i, label in enumerate(label2data.keys())}
        num_class = len(label2id)
        data_array = [(img, label2id[label]) for label, data in label2data.items() for img in data]
        return label2data, label2id, id2label, num_class, data_array

    def gen(self):
        for i in range(self.start_id, len(self.data_array), self.batch_size):
            batch = self.data_array[i:i+self.batch_size]
            imgs, ids = zip(*batch)
            yield imgs, ids

    def reset(self):
        self.start_id = 0

    def shuffle(self):
        random.shuffle(self.data_array)

    def get_num_class(self):
        return self.num_class


def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


def train():
    loader = BasicLoader(config["data_root"], config["batch_size"])
    model = make_model(input_shape=(160, 200, 3), num_classes=loader.get_num_class())
    callbacks = [
        keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
    ]
    model.compile(
        optimizer=keras.optimizers.Adam(config["lr"]),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.Accuracy()],
    )
    model.fit(
        x=loader.gen(), batch_size=config["batch_size"], epochs=config["epoch"], callbacks=callbacks, validation_data=None,
    )


def predict():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="F:/CarLogo/crawler/chelogo/")
    args = parser.parse_args()

    config["data_root"] = args.root

    train()






