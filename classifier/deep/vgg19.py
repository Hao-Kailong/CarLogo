import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np


config = {
    "data_root": "/data/klhao/CommonCar/chelogo",
    "batch_size": 256,
    "seed": 2022,
    "image_size": (224, 224),
    "input_shape": (224, 224, 3),
    "classes": 291,
    "epoch": 10,
}


# data = keras.utils.image_dataset_from_directory(
#     directory=config["data_root"],
#     labels='inferred',
#     label_mode='categorical',
#     batch_size=config["batch_size"],
#     image_size=config["image_size"],
#     shuffle=True,
#     seed=config["seed"],
# )


def one_hot(i):
    vec = np.zeros(config["classes"])
    vec[i] = 1
    return vec


def load_data(validation_split=0.2):
    label2id = {}
    i = 0
    X, Y = [], []
    for d in os.listdir(config["data_root"]):
        label2id[d] = i  # 标签到id
        dir = os.path.join(config["data_root"], d)
        for f in os.listdir(dir):
            path = os.path.join(dir, f)
            if f.endswith(".jpg") and os.path.isfile(path):
                image = plt.imread(path)
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                image = cv2.resize(image, dsize=config["image_size"])
                image = np.expand_dims(image, axis=0)
                image = preprocess_input(image)
                X.append(image)
                Y.append(one_hot(i))
        i += 1
        print("{} loaded.".format(d))
        if i == 1:
            break
    return np.vstack(X), np.asarray(Y)


def run_finetune():
    X, Y = load_data()
    base_model = VGG19(
        include_top=False,
        weights="imagenet",
        input_shape=config["input_shape"],
        pooling="max",
    )
    x = base_model.output  # (None, 512)
    output = Dense(config["classes"], activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=output)
    callbacks = [
        keras.callbacks.ModelCheckpoint("VGG19_at_{epoch}.h5"),
    ]
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics="accuracy")
    model.fit(
        x=X,
        y=Y,
        batch_size=config["batch_size"],
        epochs=config["epoch"],
        shuffle=True,
        callbacks=callbacks
    )


if __name__ == "__main__":
    run_finetune()









