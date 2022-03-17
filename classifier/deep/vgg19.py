import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input


config = {
    "data_root": "F:/Dataset/CommonCar/chelogo",
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
    model = VGG19(
        include_top=True,
        weights="imagenet",
        input_shape=config["input_shape"],
        pooling="max",
        classes=config["classes"],
        classifier_activation="softmax",
    )

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics="accuracy")
    model.fit(data, epochs=10)


if __name__ == "__main__":
    run_finetune()









