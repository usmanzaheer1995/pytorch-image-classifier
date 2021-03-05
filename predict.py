import time
import json
from pathlib import Path

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms, datasets, models
from PIL import Image

from workspace_utils import active_session
from helper import label_mapper, classifier, load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description="Imager classifier predictor")
    parser.add_argument("file_path")
    parser.add_argument("checkpoint")
    parser.add_argument("--category_names", default="cat_to_name.json")
    parser.add_argument("--top_k", default="5")
    parser.add_argument("--gpu", default="gpu")
    
    return parser.parse_args()

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    (width, height) = image.size
    
    resize = 256
    crop_size = 224
    
#     https://gist.github.com/tomvon/ae288482869b495201a0
    wpercent = (resize / float(width))
    hsize = int(float(height) * float(wpercent))
    image = image.resize((resize, hsize), Image.ANTIALIAS)
    
#     https://stackoverflow.com/questions/16646183/crop-an-image-in-the-centre-using-pil
    left = (image.size[0] - crop_size) / 2
    top = (image.size[0] - crop_size) / 2
    right = (image.size[0] + crop_size) / 2
    bottom = (image.size[0] + crop_size) / 2
    image = image.crop((left, top, right, bottom))
    
    image = np.array(image)
    image = image / 255
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    image = (image - mean) / std
    
    image = np.transpose(image, (2, 0, 1))
    return image

def predict(image_path, model, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if (device == "gpu"):
        device = "cuda"
       
    model.to(device)
    model.eval()
    
    image = Image.open(image_path)
    image = process_image(image)
    image = torch.from_numpy(image).type(torch.cuda.FloatTensor)
    
    # model expects 1D array of tensors
    image = image.unsqueeze(0)
    
    image = image.to(device)
    with torch.no_grad():
        output = model.forward(image)
    
    # https://stackoverflow.com/a/62364921/8290054
    probability = F.softmax(output.data, dim=1)
    
    top_p, top_class = probability.topk(topk)
    
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    
    classes = []
    for c in top_class[0].tolist():
        classes.append(idx_to_class[c])
    
    return top_p[0].tolist(), classes

def main():
    args = parse_args()
    model = load_checkpoint("./checkpoints/" + args.checkpoint)
    labels = label_mapper()
        
    probs, classes = predict(args.file_path, model, args.gpu, int(args.top_k))
    
    flowers = [labels[str(idx)] for idx in classes]
    
    for idx, flower in enumerate(flowers):
        print(f"{flower} with prediction of {probs[idx] * 100:.2f}%")
    
if __name__ == "__main__":
    main()


