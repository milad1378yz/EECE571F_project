import torch
from torchmetrics.image.fid import FrechetInceptionDistance
import numpy as np
import random
from torchmetrics.image.kid import KernelInceptionDistance
import os
from PIL import Image
from torchvision import transforms
import argparse


def image_loader(directory_path,  sample_size=100):
    images = []
    filenames = os.listdir(directory_path)
    random.shuffle(filenames)
    for filename in filenames[:sample_size]:
        image = Image.open(os.path.join(directory_path, filename))
        image = image.convert('RGB')
        image = transforms.Resize((128, 128))(image)
        image = np.array(image)
        images.append(image)
    images = np.array(images)
    
    images = np.transpose(images, (0, 3, 1, 2))
    image_tensor = torch.tensor(images, dtype=torch.uint8)
    return image_tensor
# set the seed of random number generator



random.seed(0)
parser = argparse.ArgumentParser()
parser.add_argument('--dir_generator', type=str, help='Directory path for data generator',default='/scratch/st-sdena-1/miladyz/EECE571F_project/text2image/unconditional/generated/')
parser.add_argument('--dir_separated_data', type=str, help='Directory path for separated data',default='/scratch/st-sdena-1/miladyz/EECE571F_project/data/seprated_data/')
args = parser.parse_args()

dir_generator = args.dir_generator
dir_separated_data = args.dir_separated_data

for filename in os.listdir(dir_generator):
    print(filename)
    data_dir = os.path.join(dir_separated_data, filename)
    data1 = image_loader(data_dir, sample_size=100)
    data_dir = dir_generator+filename
    data2 = image_loader(data_dir, sample_size=100)

    fid = FrechetInceptionDistance(feature=64)
    fid.update(data1, real=True)
    fid.update(data2, real=False)
    print("FID:",fid.compute())

    kid = KernelInceptionDistance(subset_size=50)

    kid.update(data1, real=True)
    kid.update(data2, real=False)
    kid_mean, kid_std = kid.compute()
    print("KID",(kid_mean, kid_std))





