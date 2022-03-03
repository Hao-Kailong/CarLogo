import albumentations as A
import cv2
import os
import random
import matplotlib.pyplot as plt


config = {
    "white": [255, 255, 255],
}


def gen_id():
    return "".join([random.choice("0123456789abcdefghijklmnopqrstuvwxyz") for i in range(8)])


def save(array, raw_path):
    path, filename = os.path.split(raw_path)
    name, suffix = os.path.splitext(filename)
    for a in array:
        id = gen_id()
        saved_path = os.path.join(path, name + "-albumentations-" + id + ".jpg")
        plt.imsave(saved_path, a)


def augment(path):
    image = plt.imread(path)
    padding = config["white"]

    capricious = A.Compose([
        A.Flip(p=0.5),
        A.Rotate(limit=180, border_mode=cv2.BORDER_CONSTANT, value=padding, p=0.5),
        A.Affine(shear=(-20, 20), cval=padding, p=0.5),
        A.Perspective(pad_mode=cv2.BORDER_CONSTANT, pad_val=padding, p=0.5),
        # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        # A.RandomRain(p=0.5),
        A.Emboss(p=0.5),
        A.Equalize(p=0.5),
    ])
    results = []
    for i in range(100):
        one = capricious(image=image)["image"]
        results.append(one)
    save(results, path)


def main(root):
    for d in os.listdir(root):
        d = os.path.join(root, d)
        if not os.path.isdir(d):  # 文件夹
            continue
        for f in os.listdir(d):
            f = os.path.join(d, f)
            if os.path.isfile(f) and f.endswith(".jpg") and "albumentations" not in f:  # 图片
                augment(f)
                print("{} processed.".format(f))


def delete(root):
    for d in os.listdir(root):
        d = os.path.join(root, d)
        if not os.path.isdir(d):  # 文件夹
            continue
        for f in os.listdir(d):
            f = os.path.join(d, f)
            if os.path.isfile(f) and f.endswith(".jpg") and "albumentations" in f:  # 图片
                os.remove(f)


if __name__ == "__main__":
    main("C:/Users/Dell/Desktop/trail")
    # delete("C:/Users/Dell/Desktop/trail")
