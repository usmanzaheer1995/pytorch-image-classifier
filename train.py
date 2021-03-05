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
from helper import label_mapper, classifier, save_checkpoint, load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description="Imager classifier trainer")
    parser.add_argument("data_dir")
    parser.add_argument("--save_dir", default="./checkpoints/")
    parser.add_argument("--arch", default="vgg16", choices=["vgg16", "densenet121", "resnet50"])
    parser.add_argument("--dropout", default="0.5")
    parser.add_argument("--learning_rate", default="0.001")
    parser.add_argument("--hidden_units", default="512")
    parser.add_argument("--epochs", default="5")
    parser.add_argument("--gpu", default="gpu")
    
    return parser.parse_args()

def transformers(args):
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    training_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                             transforms.RandomRotation(30),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406],
                                                                  [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(255),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    image_datasets = {
        "train": datasets.ImageFolder(train_dir, transform=training_transforms),
        "test": datasets.ImageFolder(test_dir, transform=test_transforms),
        "validate": datasets.ImageFolder(valid_dir, transform=validation_transforms)
    }

    dataloaders = {
        "train": torch.utils.data.DataLoader(image_datasets["train"], batch_size=64, shuffle=True),
        "test": torch.utils.data.DataLoader(image_datasets["test"], batch_size=64, shuffle=True),
        "validate": torch.utils.data.DataLoader(image_datasets["validate"], batch_size=64, shuffle=True)
    }
    
    return image_datasets, dataloaders

def train(model, criterion, optimizer, trainloader, validationloader, epochs, device):
	if (device == "gpu"):
            device = "cuda"
        print(f"Training {model} model...")
        steps = 0
        running_loss = 0
        print_every = 10
        for epoch in range(epochs):
            for inputs, labels in trainloader:
                steps += 1
                # Move input and label tensors to the default device
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                logps = model.forward(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    test_loss = 0
                    accuracy = 0
                    model.eval()
                    with torch.no_grad():
                        for inputs, labels in validationloader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            logps = model.forward(inputs)
                            batch_loss = criterion(logps, labels)

                            test_loss += batch_loss.item()

                            # Calculate accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Validation loss: {test_loss/len(validationloader):.3f}.. "
                          f"Validation accuracy: {accuracy/len(validationloader):.3f}")
                    running_loss = 0
                    model.train()
    print("Training complete!")
        
    
def test_network(model, test_loader, criterion, device):
    print("Testing model accuracy...")
    device = "cuda" if device == "gpu" else "cpu"
    test_loss = 0
    accuracy = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            log_ps = model.forward(inputs)
            batch_loss = criterion(log_ps, labels)
            test_loss += batch_loss.item()

            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print(f"Test accuracy: {accuracy/len(test_loader):.3f}..")

def main():
    args = parse_args()

    # if save_dir does not exist, create the directory
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    image_datasets, dataloaders = transformers(args)
    flowers_to_name = label_mapper()
    
    model, criterion, optimizer = classifier(args.arch,
                                             float(args.dropout),
                                             int(args.hidden_units),
                                             float(args.learning_rate),
                                             args.gpu)
    
    model.class_to_idx = image_datasets['train'].class_to_idx
    train(model, criterion, optimizer, dataloaders["train"], dataloaders["validate"], int(args.epochs), args.gpu)
    test_network(model, dataloaders["test"], criterion, args.gpu)
    save_checkpoint(args.save_dir, model, optimizer, args.arch, int(args.epochs), float(args.learning_rate))

if __name__ == "__main__":
    main()

