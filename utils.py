import os
import typing

import cv2  # pytype: disable=attribute-error
import matplotlib
import numpy as np
import torch
import tqdm
import pandas as pd
from PIL import Image
from sklearn.metrics import r2_score
import torchvision.transforms as T


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


def eval_single(model,img):
    """ evaluates single img and returns as a numpy array"""
    transform_input = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = transform_input(img)[None,]
    with torch.no_grad():
        output = model(img)
    output = model(img)
    return (output[0]>0.).float().detach().numpy()

def output_to_PIL(output):
    return Image.fromarray(output[0].astype('uint8')*255)
# make this more efficient
def get_FAC_R2(model):

    folder_path="/workspace/data/NAS/RV-Milos/RV_images/"
    df = pd.read_csv(f"{folder_path}/VideoList.csv")
    # get relevant frames
    pair_frames = []
    for i, row in df.iterrows():
        if len(df['Tracings'][i].split("'"))==5:
            name = df['FileName'][i]
            frame1 = df['Tracings'][i].split("'")[1]
            frame2 = df['Tracings'][i].split("'")[3]
            pair_frames.append([ f"{name}_{frame1}" , f"{name}_{frame2}"])

    # get real FAC
    real_FAC = []
    for pair in pair_frames:
        area1 = np.sum((np.array(Image.open(f"{folder_path}/Masks/{pair[0]}.png"))>0).astype('uint8'))
        area2 = np.sum((np.array(Image.open(f"{folder_path}/Masks/{pair[1]}.png"))>0).astype('uint8'))
        real_FAC.append((max(area1,area2)-min(area1,area2))/max(area1,area2))

    pred_FAC = []
    for pair in pair_frames:
        input1 = Image.open(f"{folder_path}/Images/{pair[0]}.png")
        input2 = Image.open(f"{folder_path}/Images/{pair[1]}.png")
        area1 = np.sum(eval_single(model,input1))
        area2 = np.sum(eval_single(model,input2))
        pred_FAC.append( (max(area1,area2)-min(area1,area2))/max(area1,area2) ) 

    real_FAC = np.array(real_FAC)
    pred_FAC = np.array(pred_FAC)
    return r2_score(real_FAC,pred_FAC)
