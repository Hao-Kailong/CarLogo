# https://keras.io/examples/vision/image_classification_from_scratch/
# Xception Model
from tensorflow import keras
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from collections import defaultdict
import argparse
import numpy as np


config = {
    "batch_size": 32,
    "epoch": 50,
    "lr": 1e-3,
    "saved_model": "save_at_50.h5",
}


class BasicLoader(object):
    def __init__(self, root, batch_size):
        self.root = root
        self.batch_size = batch_size
        self.start_id = 0
        self.label2data, self.label2id, self.id2label, self.num_class, self.X, self.Y = self.load()
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
                    if image.shape != (160, 200, 3):
                        image = cv2.resize(image, dsize=(200, 160))
                    assert image.shape == (160, 200, 3)
                    label2data[d].append(image)
        label2id = {label: i for i, label in enumerate(label2data.keys())}
        id2label = {i: label for i, label in enumerate(label2data.keys())}
        num_class = len(label2id)
        X = [img for label, data in label2data.items() for img in data]
        Y = [label2id[label] for label, data in label2data.items() for img in data]
        return label2data, label2id, id2label, num_class, X, Y

    def get_num_class(self):
        return self.num_class

    def get_X(self):
        return np.asarray(self.X)

    def get_Y(self):
        return keras.utils.to_categorical(np.asarray(self.Y), num_classes=self.num_class)

    def get_len(self):
        return len(self.X)


def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = keras.layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    x = keras.layers.Conv2D(64, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.SeparableConv2D(size, 3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Activation("relu")(x)
        x = keras.layers.SeparableConv2D(size, 3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = keras.layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = keras.layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = keras.layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    x = keras.layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


def train():
    loader = BasicLoader(config["data_root"], config["batch_size"])
    model = make_model(input_shape=(160, 200, 3), num_classes=loader.get_num_class())
    callbacks = [
        keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
    ]
    model.compile(
        optimizer=keras.optimizers.Adam(config["lr"]),
        loss="categorical_crossentropy",
        metrics=[tf.keras.metrics.Accuracy()],
    )
    steps_per_epoch = loader.get_len() // config["batch_size"]
    model.fit(
        x=loader.get_X(),
        y=loader.get_Y(),
        batch_size=config["batch_size"],
        shuffle=True,
        epochs=config["epoch"],
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
    )


def predict(img):
    model = keras.models.load_model(config["saved_model"])
    try:
        img = cv2.resize(img, (200, 160))
    except Exception as e:
        print(e)
    result = model.predict(img)
    return result


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--root", default="F:/CarLogo/crawler/chelogo/")
    # args = parser.parse_args()
    #
    # config["data_root"] = args.root
    #
    # train()

    img = plt.imread("/data/klhao/CarLogo/crawler/chelogo/宝马/logo.jpg")
    result = predict(img)
    print(result)

