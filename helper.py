import json
import torch
from torch import nn, optim
from torchvision import models

def label_mapper():
    with open('cat_to_name.json', 'r') as f:
        return json.load(f)
    
def save_checkpoint(save_dir, model, optimizer, arch, epochs, lr):
    num_features = model.fc[0].in_features if arch == "resnet50" else model.classifier[0].in_features
    checkpoint = {
        "model_pretrained": arch,
        "input_size": num_features,
        "output_size": 102,
        "state_dict": model.state_dict(),
        "class_to_idx": model.class_to_idx,
        "optimizer": optimizer.state_dict(),
        "epochs": epochs,
        "learning_rate": lr,
    }
    checkpoint["classifier"] = model.fc if arch == "resnet50" else model.classifier
    checkpoint_dir = save_dir + arch + "_checkpoint.pth"
    torch.save(checkpoint, checkpoint_dir)
    
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = getattr(models, checkpoint["model_pretrained"])(pretrained=True)
    model.epochs = checkpoint["epochs"]
    model.class_to_idx = checkpoint["class_to_idx"]
    model.optimizer = checkpoint["optimizer"]
    model.learning_rate = checkpoint["learning_rate"]
    
    if (checkpoint["model_pretrained"] == "resnet50"):
        model.fc = checkpoint["classifier"]
    else:
        model.classifier = checkpoint["classifier"]
    
    model.load_state_dict(checkpoint['state_dict'])
    
    return model
    
def classifier(arch, dropout, hidden_units, lr, gpu):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if (gpu == "gpu"):
        device = "cuda"
    model = getattr(models, arch)(pretrained=True)
    
    for param in model.parameters():
        param.require_grad = False
    
    num_features = 0
    if (arch == "resnet50"):
        num_features = model.fc.in_features
    else:
        num_features = model.classifier[0].in_features if arch == "vgg16" else model.classifier.in_features
#     model.classifier = nn.Sequential(nn.Linear(num_features, hidden_units),
#                                      nn.ReLU(),
#                                      nn.Dropout(dropout),
#                                      nn.Linear(hidden_units, 102),
#                                      nn.LogSoftmax(dim=1))

#     Without another hidden layer, vgg16 model was not performing
#     as well as I hoped on the test data (even though accuracy on 
#     validation data was above 80% during training).
#     So I opted for this instead of above
    if (arch == "vgg16"):
        model.classifier = nn.Sequential(nn.Linear(num_features, 2048),
                                         nn.ReLU(),
                                         nn.Linear(2048, 500),
                                         nn.ReLU(),
                                         nn.Linear(500, 112),
                                         nn.LogSoftmax(dim=1))
    if  (arch == "densenet121"):
        model.classifier = nn.Sequential(nn.Linear(num_features, hidden_units),
                                         nn.ReLU(),
                                         nn.Dropout(dropout),
                                         nn.Linear(hidden_units, 102),
                                         nn.LogSoftmax(dim=1))
    if (arch == "resnet50"):
#         trained with dropout=0.2
        model.fc = nn.Sequential(nn.Linear(num_features, hidden_units),
                                   nn.ReLU(),
                                   nn.Dropout(dropout),
                                   nn.Linear(hidden_units, 102),
                                   nn.LogSoftmax(dim=1))
    criterion = nn.NLLLoss()
    
    if (arch == "resnet50"):
        optimizer = optim.Adam(model.fc.parameters(), lr=lr)
    else:
        optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

    model.to(device)
    return model, criterion, optimizer
    