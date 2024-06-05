from ultralytics import YOLO
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid', font_scale=1.3, palette='Set2')

import torch
#import torchvision
from torchvision import transforms

def add_margin(pil_img, new_width, new_height):
    width, height = pil_img.size
    left = (new_width - width) // 2
    top = (new_height - height) // 2
    result = Image.new(pil_img.mode, (new_width, new_height), (255, 255, 255))
    result.paste(pil_img, (left, top))
    return result


if __name__ == '__main__':
    model = YOLO('./best.pt')
    img = Image.open('QR330.png')
    sz = []
    conf = []
    for size in tqdm(range(640, 4001)):
        sz.append(300 * (640 / size))
        expand_img = add_margin(img, size, size)
        conf_ = []
        for _ in range(15):
            transforms.RandomRotation((-180, 180))(expand_img).save('temp.png')
            result = model(['temp.png'])
            if len(result) == 0 or len(result[0].boxes.conf) == 0:
                conf_.append(0)
            else:
                conf_.append(result[0].boxes.conf[0].item())
        conf.append(np.mean(conf_))

    plt.title("Изменение confidence предсказания модели при уменьшении размера QR-кода")
    plt.plot(sz, conf)
    plt.xlabel("Размер")
    plt.ylabel("Confidence")
    plt.show()