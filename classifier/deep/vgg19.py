import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model


config = {
    "data_root": "/data/klhao/CommonCar/chelogo",
    "batch_size": 256,
    "seed": 2022,
    "image_size": (224, 224),
    "input_shape": (224, 224, 3),
    "classes": 291,
}


data = keras.utils.image_dataset_from_directory(
    directory=config["data_root"],
    labels='inferred',
    label_mode='categorical',
    batch_size=config["batch_size"],
    image_size=config["image_size"],
    shuffle=True,
    seed=2022,
)


def run_finetune():
    base_model = VGG19(
        include_top=False,
        weights="imagenet",
        input_shape=config["input_shape"],
        pooling="max",
    )
    x = base_model.output  # (None, 512)
    output = Dense(config["classes"], activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics="accuracy")
    model.fit(data, epochs=10)


if __name__ == "__main__":
    run_finetune()









