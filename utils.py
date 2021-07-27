import os
import typing

import cv2  # pytype: disable=attribute-error
import matplotlib
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
from PIL import Image
from sklearn.metrics import r2_score
import torchvision.transforms as T
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import animation,rc


def load_video(filename: str) -> np.ndarray:
    """Loads a video from a file.

    Args:
        filename (str): filename of video

    Returns:
        A np.ndarray with dimensions (channels=3, frames, height, width). The
        values will be uint8's ranging from 0 to 255.

    Raises:
        FileNotFoundError: Could not find `filename`
        ValueError: An error occurred while reading the video
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    capture = cv2.VideoCapture(filename)

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    v = np.zeros((frame_count, frame_width, frame_height, 3), np.uint8)

    for count in range(frame_count):
        ret, frame = capture.read()
        if not ret:
            raise ValueError("Failed to load frame #{} of {}.".format(count, filename))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        v[count] = frame

    return v

def dice_loss(input, target):
    return 1 - dice_metric(input,target)

def dice_metric(input,target):
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

def augment(image,mask):
    # select angle in range (-20,20)
    angle = np.random.randint(40)-20
    # select translate x in range (-0.1,0.1) times image width
    translate_x = (np.random.rand()-0.5)/5 * mask.size[0]
    # select translate y in range (-0.1,0.1) times image height
    translate_y = (np.random.rand()-0.5)/5 * mask.size[1]
    # selectscale in range (0.9,1.1)
    scale = 1+(np.random.rand()-0.5)/5
    # select shear in range (-10,10)
    shear = np.random.randint(20)-10	

    # apply affine transform
    image = T.functional.affine(image,angle=angle,translate=(translate_x,translate_y),scale=scale,shear=shear)
    mask = T.functional.affine(mask,angle=angle,translate=(translate_x,translate_y),scale=scale,shear=shear)

    # probabilities for additional augmentations
    ps = np.array([0.15,0.15,0.15,0.2])
    ps=(np.random.rand(4)<ps).astype(float)
    # ps contains only zeros and ones
    # the reason we did it this way is because we need the same augmentations for both image and mask
    transform = T.Compose([
        T.RandomApply(transforms=[T.Pad(padding=5)], p=ps[0]),
        T.RandomApply(transforms=[T.Pad(padding=4)], p=ps[1]),
        T.RandomApply(transforms=[T.Pad(padding=3)], p=ps[2]),
        T.RandomApply(transforms=[T.CenterCrop((100,100))], p=ps[3])
    ])

    image = transform(image)
    mask = transform(mask)
    return image,mask

def evaluate_FAC(model,dataloader):

    device=torch.device('cuda')
    pred_fac = []
    real_fac = []
    with torch.no_grad():
        for frame1, frame2, ef in tqdm(dataloader):
            frame1 = frame1.to(device)
            frame2 = frame2.to(device)

            out1 = model(frame1)['out'].float().detach().cpu().numpy()
            out2 = model(frame2)['out'].float().detach().cpu().numpy()

            areas1 = []
            areas2 = []
            fac = []

            areas1 = np.sum(np.sum(out1>0.,axis=2),axis=2).reshape(-1)
            areas2 = np.sum(np.sum(out2>0.,axis=2),axis=2).reshape(-1)
            fac = np.abs(areas1-areas2)/np.maximum(areas1,areas2)

            pred_fac = np.concatenate((pred_fac,fac))
            real_fac = np.concatenate((real_fac,ef))
        
        sklearn_r2=0.0
        r2=0.0

        try:
            sklearn_r2 = r2_score(real_fac,pred_fac)
        except:
            print()
        try:
            r2 = stats.linregress(real_fac,pred_fac).rvalue**2
        except:
            print()

        print(f"R2: {sklearn_r2}")
        print(f"R2: {r2}")
        return sklearn_r2,r2

def create_video(video):
    """ Takes a numpy array and returns animation"""
    fig, ax = plt.subplots()
    plt.close()
    def animator(N): # N is the animation frame number
        ax.imshow(video[N])
        return ax
    PlotFrames = range(0,video.shape[0],1)
    anim = animation.FuncAnimation(fig,animator,frames=PlotFrames,interval=100)
    rc('animation', html='jshtml')
    return anim 