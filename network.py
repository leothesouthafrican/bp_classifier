from torchvision import models
from image_handler import ImageEditor
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
import torch
import os

model = models.segmentation.fcn_resnet101(pretrained=True).eval().to('mps')

images = ImageEditor('images', 'edited_images')

#pass each image in the edited_images folder through the model 
for image in tqdm(os.listdir('edited_images'), desc='Segmenting Images'):
    print('edited_images/' + image)
    images.segment(model,'edited_images/' + image)











