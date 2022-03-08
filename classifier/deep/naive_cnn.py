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
    "data_root": "F:/CarLogo/crawler/chelogo/",
    "batch_size": 300,
    "epoch": 50,
    "lr": 1e-4,
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
    try:
        model = keras.models.load_model(config["saved_model"])
    except:
        pass
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


def predict(img_path):
    model = keras.models.load_model(config["saved_model"])
    try:
        img = plt.imread(img_path)
        if len(img) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if img.shape != (160, 200, 3):
            img = cv2.resize(img, dsize=(200, 160))
    except Exception as e:
        print(e)
        return []
    logits = model.predict(np.asarray([img]))
    id2label = get_label()
    result = [id2label[i] for i in np.argmax(logits, axis=1)]
    return result


def get_label():
    id2label = {}
    i = 0
    root = config["data_root"]
    for d in os.listdir(root):
        sub_dir = os.path.join(root, d)
        if os.path.isfile(sub_dir):
            continue
        id2label[i] = d
        i += 1
    return id2label


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="")
    parser.add_argument("--mode", default="train", choices=["train", "predict"])
    parser.add_argument("--img", type=str)
    args = parser.parse_args()

    if args.root:
        config["data_root"] = args.root

    if args.mode == "train":
        train()
    else:
        result = predict(args.img)
        print(result)

