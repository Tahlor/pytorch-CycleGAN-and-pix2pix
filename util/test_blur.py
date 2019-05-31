import collections
import os
import torch
from scipy.ndimage.filters import gaussian_filter
import numpy as np
import util
from PIL import Image
from data.base_dataset import BaseDataset, get_transform
import matplotlib.pyplot as plt

def mask(fake_img, gt, sigma=10, no_mask=False, cuda=False):  # .5 ~ 3 pixles, 1 ~ 5 pixels
    if no_mask:
        return fake_img
    else:
        #  + 1) / 2.0 * 255.0
        # Colors normalized to -1,1; -1 = BLACK, 1 = WHITE

        white_space = (gaussian_filter(gt, sigma=sigma) > 0)
        white_space = torch.from_numpy(np.array(2 * white_space, dtype=np.float32))
        # fake_img[~blurred] = -1
        if cuda:
            white_space = white_space.to("cuda")
        #fake_img = fake_img + (white_space)  # add one to everything over 0 (everything light gray)
        fake_img = fake_img * (white_space-2)*-1/2 #
        # print(fake_img)
        return fake_img


def load_image(path,input_nc=1, output_nc=1):
    A_img = Image.open(path).convert("RGB")
    Opt = collections.namedtuple("Opt", ["resize_or_crop", "loadSizeY", "loadSizeX", "fineSizeY", "fineSizeX", "isTrain"], verbose=False, rename=False)
    opt = Opt(resize_or_crop = "resize_and_crop", loadSizeY=64, loadSizeX=1920, fineSizeY=64, fineSizeX=1920, isTrain=False)

    transform = get_transform(opt)
    A = transform(A_img)

    if input_nc == 1:  # RGB to gray
        tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
        A = tmp.unsqueeze(0)
    return A

def save(image_numpy, image_path):
    util.save_image(image_numpy, image_path)

def image2tensor():
    pass


def display(img):
    # img = img.permute(1,2,0).squeeze()
    # print(img.shape)
    # plt.imshow(img)
    img = util.tensor2im(img, batch=False)
    plt.interactive(False)

    plt.imshow(img)
    plt.show()

def loop():
    pass

if __name__ == "__main__":
    path = "/media/SuperComputerGroups/fslg_hwr/hw_data/handwriting/train_online_original"
    img_list = os.listdir(path)
    img_path = os.path.join(path, img_list[0])
    print(img_path)
    img = load_image(img_path)
    display(mask(img, img))