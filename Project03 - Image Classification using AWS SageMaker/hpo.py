#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import logging

import argparse

from smdebug import modes
from smdebug.profiler.utils import str2bool
import json
import os
import time
import smdebug.pytorch as smd
from smdebug.pytorch import get_hook

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True    # Load image even if it's truncated.

def test(model, test_loader, criterion, device):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()         # Set model.eval() when we are not on training phase
    with torch.no_grad():
        running_loss=0
        running_corrects=0
        running_corrects=0
        running_samples=0

        for inputs, labels in test_loader:
            inputs=inputs.to(device)
            labels=labels.to(device)
            outputs=model(inputs)
            
            loss=criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            running_samples += len(inputs)

        total_loss = running_loss / len(test_loader.dataset)
        total_acc = running_corrects/ len(test_loader.dataset)
        
        print(f"Data set size: {len(test_loader.dataset)}, running samples: {running_samples}")
        print(
            "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            total_loss, running_corrects, running_samples, 100.0 * total_acc)
        )  
    pass

def train(model, train_loader, valid_loader, criterion, optimizer, epochs, device):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.train()
    
    for i in range(epochs):
        print("START TRAINING")
        model.train()
        train_loss = 0
        running_samples = 0
        for image_no, (inputs, targets) in enumerate(train_loader):
            if (image_no % 100 == 0):
                print('fitting {}'.format(image_no))
            inputs, targets = inputs.to(device), targets.to(device)
            model = model.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            running_samples+=len(inputs)
        print(f"Data set size: {len(train_loader.dataset)}, running samples: {running_samples}")
            #NOTE: train on smaller sample since time is running low
#             if running_samples>(0.2*len(train_loader.dataset)):
#                 break

        print("START VALIDATING")
        #TODO: Set hook to eval mode
        model.eval()
        with torch.no_grad():
            val_loss = 0
            with torch.no_grad():
                for _, (inputs, targets) in enumerate(valid_loader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            print(
                "Epoch %d: train loss %.3f, val loss %.3f"
                % (i, train_loss, val_loss)
            )
    return model

def net(model):
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.__dict__[model](pretrained=True)
    for param in model.parameters():
        param.requires_grad = False   

    num_features = model.fc.in_features
    num_classes = 133
#     model.fc = nn.Linear(num_features, 133)
    model.fc = nn.Sequential(nn.Linear(num_features,512),
                                  nn.ReLU(),
                                  nn.Dropout(),
                                  nn.Linear(512, num_classes))
    
    # check if CUDA is available
    use_cuda = torch.cuda.is_available()
    
    # move model to GPU if CUDA is available
    if use_cuda:
        model = model.cuda()
        print("GPU available for training")

    return model

def create_data_loaders(data, batch_size, transformer, is_shuffle = False):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    import torchvision.datasets as datasets
    
    image_data = datasets.ImageFolder(data, transform=transformer)
    
    print(len(image_data.classes))
    
    return  torch.utils.data.DataLoader(image_data, batch_size=batch_size, shuffle=is_shuffle)

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net(args.model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=model.to(device)
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    img_dimension = 224
    
    train_transformer = transforms.Compose([
        transforms.Resize(img_dimension),
        transforms.CenterCrop(img_dimension),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_loader = create_data_loaders(args.train, args.batch_size, train_transformer, True)
    
    test_transformer = transforms.Compose([
        transforms.Resize(img_dimension),
        transforms.CenterCrop(img_dimension),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    valid_loader = create_data_loaders(args.validation, args.batch_size, test_transformer)
    
    model=train(model, train_loader, valid_loader, loss_criterion, optimizer, args.epochs, device)

    
    '''
    TODO: Test the model to see its accuracy
    '''
    test_loader = create_data_loaders(args.test, args.batch_size, test_transformer)
    
    test(model, test_loader, loss_criterion, device)
    
    '''
    TODO: Save the trained model
    '''
    path = os.path.join(args.model_dir, "model.pth")
    torch.save(model.to(device).state_dict(), path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        metavar="N",
        help="input batch size for training (default: 100)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    
    parser.add_argument(
        "--lr", 
        type=float, 
        default=0.01, 
        metavar="LR", 
        help="learning rate (default: 0.01)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18"
    )

    parser.add_argument(
        "--train",
        type=str,
        default=os.environ["SM_CHANNEL_TRAIN"],
        help="training data input (default: set in env variables)"
    )

    parser.add_argument(
        "--test",
        type=str,
        default=os.environ['SM_CHANNEL_TEST'],
        help="test data input (default: set in env variables)"
    )
    
    parser.add_argument(
        "--validation",
        type=str,
        default=os.environ['SM_CHANNEL_VALIDATION'],
        help="validation data input (default: set in env variables)"
    )    
    
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ["SM_MODEL_DIR"],
        help="directory to save model data (default: set in env variables)"
    )
    
    args=parser.parse_args()
    #args = vars(parser.parse_args())

    main(args)
