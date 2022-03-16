import os
import matplotlib.pyplot as plt
# plt.rcParams["font.sans-serif"] = ["SimHei"]


config = {
    "root": "F:/Dataset/CommonCar/chelogo"
}


def sample_count():
    label2count = {}
    for d in os.listdir(config["root"]):
        images = [f for f in os.listdir(os.path.join(config["root"], d)) if f.endswith(".jpg")]
        label2count[d] = len(images)
    label2count = {label: count for label, count in sorted(label2count.items(), key=lambda x: x[1], reverse=True)}
    plt.bar(x=range(len(label2count)), height=label2count.values())
    plt.show()


if __name__ == "__main__":
    sample_count()

