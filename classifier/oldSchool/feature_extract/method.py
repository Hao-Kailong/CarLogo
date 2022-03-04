import matplotlib.pyplot as plt
import skimage
import os
import cv2
from skimage.feature import local_binary_pattern
from skimage.filters import sobel


config = {
    "dir": "./image",
    "radius": 1,
    "n_points": 8,
}


def LBP(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_lbp = local_binary_pattern(img, P=config["n_points"], R=config["radius"], method="var")
    cv2.imwrite(image_path + ".lbp.jpg", img_lbp)


def SOBEL(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_lbp = sobel(img)
    cv2.imwrite(image_path + ".sobel.jpg", img_lbp)


def main():
    for file in os.listdir(config["dir"]):
        if file.endswith(".jpg"):
            LBP(os.path.join(config["dir"], file))
            print(file + " is processed.")


def delete():
    for file in os.listdir(config["dir"]):
        if file.endswith(".lbp.jpg"):
            os.remove(os.path.join(config["dir"], file))


if __name__ == "__main__":
    main()
    # delete()









