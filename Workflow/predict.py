from skimage.filters import gaussian as gaussian2,laplace as laplace2
import numpy as cp
import scipy.ndimage as ndimage
import skimage.exposure as exposure2
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model import FinalCapsNet

valid_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(),
    transforms.Resize((224, 224)),
    #ToArray(to_array = 'cupy'),
    #Enhance(),
])

def unsharp_mask(image):
    # Calculate the high-pass filtered image (unsharp mask)
    high_pass = laplace2(gaussian2(image, sigma=5, mode='reflect'))

    # Apply a threshold to highlight only weaker edges
    high_pass_thresholded = cp.where((high_pass >= 0), high_pass, cp.mean(high_pass))

    # Normalize the values to the range [0, 255]
    sharpened_image = (high_pass_thresholded - cp.min(high_pass_thresholded)) * 255 / (cp.max(high_pass_thresholded) - cp.min(high_pass_thresholded))

    # Clip the values to ensure they are within the valid range [0, 255]
    sharpened_image = cp.clip(sharpened_image, 0, 255)

    return sharpened_image.astype(cp.uint8)

def image_preprocessing(image,step_1 = False):
    
    image = valid_transforms(image)
    image  = image *255

    # Gaussian filtering with different sigma values
    gaussian_50 = cp.clip(cp.clip(image - gaussian2(image, sigma = 50, mode = 'reflect'), 0, 255), 0, 255)
    gaussian_150 = cp.clip(image - gaussian2(image, sigma = 150, mode = 'reflect'), 0, 255)
    gaussian_200 = cp.clip(image - gaussian2(image, sigma = 200, mode = 'reflect'), 0, 255)



    # Kernel sharpening for intermediate strength edges
    if step_1:
      high_pass_image = unsharp_mask(image)
      high_pass_image = exposure2.match_histograms(high_pass_image)

    # Combine sharpened Gaussian filtered images

      combined_image = (gaussian_50 +
                      gaussian_150 + gaussian_200 + high_pass_image)/4/255
    else:
      combined_image =(gaussian_50 +
                      gaussian_150 + gaussian_200) / 3/255
  
    



    return combined_image
def predicter(image):
    test = Image.open(r"reference_images/image2.png")
    ref= Image.open(r"reference_images\Reference_bone_image.jpg")
    test1= image_preprocessing(image)

    model=torch.load(r"model\ECN_MURA.pth",map_location=torch.device('cpu'))
    ECN = FinalCapsNet()
    ECN.load_state_dict(torch.load(r"model\ECN_MURA.pth",map_location=torch.device('cpu')))

    ECN.eval()
    test1= test1.unsqueeze(0)
    X1, y1 = ECN(test1, mode='eval')
                
    y1 = y1 / torch.sum(y1, dim=1, keepdim=True)


    predictions = int(torch.argmax(y1, dim=1))
    if predictions==1:
       return "Normal"
    elif predictions ==0:
       return "abnormal"
    else:
       return "Error"

