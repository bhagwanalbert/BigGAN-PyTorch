import os
import numpy as np
from PIL import Image
from data_loader import CORE50

dataset = CORE50(root='/home/abhagwan/datasets/core50', scenario="nicv2_391")

new_root = "/home/abhagwan/BigGAN-PyTorch/data/core50"

train_x, train_y = next(iter(dataset))
train_x = train_x.astype(np.uint8)
train_y = train_y // 5

for f in range(int(np.max(train_y))+1):
    try:
        os.mkdir(new_root + "/c" + format(f,"02d"))
    except FileExistsError:
        pass
    else:
        print("Some error occured while creating folders")

for i in range(train_y.shape[0]):

    im = Image.fromarray(train_x[i])
    im.save(new_root + "/c" + format(int(train_y[i]),"02d") + "/C_" + format(i,"04d") + ".png")
