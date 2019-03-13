#!/usr/bin/env python3

import numpy as np

import torch
from torch import nn
from torch import optim
#import torch.nn.functional as F
from torchvision import datasets, transforms

import make_model

import os
import json

# Defining command line arguments for input
import argparse
parser = argparse.ArgumentParser(add_help=True,
    description='This is flower species prediction model training program'
)
parser.add_argument('data_dir', action="store", default="flowers",
                    help="Image files directory path")
parser.add_argument('--save_dir', action="store", dest="save_dir", default="",
                    help="Model - Checkpoint save location")
parser.add_argument('--save_name', action="store", dest="save_name", default="checkpoint.pth",
                    help="Model - Checkpoint save location")
parser.add_argument('--arch', action="store", dest="arch", default="vgg16",
                    help="Pre-defined model")
parser.add_argument('--learning_rate', action="store", dest="lr", default=0.001, type=float,
                    help="Model - Learning rate")
parser.add_argument('--drop_p', action="store", dest="drop_p", default=0.2, type=float,
                    help="Model - Drop Probability")
parser.add_argument('--hidden_units', nargs="+", action="store", dest="hidden_units", default=[], type=int,
                    help="Model - Hidden units <int> <int> ...")
parser.add_argument('--epochs', action="store", dest="epochs", default=5, type=int,
                    help="Model - epochs")
parser.add_argument('--gpu', action="store_true", dest="useOnlyGPU", default=False,
                    help="Should the model untilize only GPU?")

# Assigning input to variables
inputs = parser.parse_args()
data_dir = inputs.data_dir
save_dir = inputs.save_dir
save_name = inputs.save_name
arch = inputs.arch
lr = inputs.lr
drop_p = inputs.drop_p
hidden_size = inputs.hidden_units
epochs = inputs.epochs
useOnlyGPU = inputs.useOnlyGPU

print_every = 35
input_size = 0
output_size = 102

# Printing out inputs to check
print(inputs)

# If save_dir doesn't exist, create the path
if save_dir != '':
    if not save_dir.endswith('/'):
        save_dir += '/'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

# Assigning Device: Chooses GPU if available unless specified GPU only
device = torch.device("cpu")
if(useOnlyGPU == True):
    if(torch.cuda.is_available()):
        device = torch.device("cuda:0")
    else:
        print("Error: GPU not available")
        exit(1)
elif(torch.cuda.is_available()):
    device = torch.device("cuda:0")

# Folders
#data_dir  = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir  = data_dir + '/test'

# Transforms for the training, validation, and testing sets
train_transform = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

valid_transform = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

test_transform  = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

# Loading datasets with ImageFolder
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transform)
test_dataset  = datasets.ImageFolder(test_dir,  transform=test_transform)

# Defining DataLoaders
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=16, shuffle=False)
testloader  = torch.utils.data.DataLoader(test_dataset,  batch_size=16, shuffle=False)

# Import flower species names
#with open('cat_to_name.json', 'r') as f:
#    cat_to_name = json.load(f)

#_______________________________________________________________________________________________________
########################################################################################################

# Validation function
def validation(p_model, p_someloader, p_criterion, p_device):
    l_test_loss = 0
    l_accuracy = 0
    for l_images, l_labels in p_someloader:
        
        l_images, l_labels = l_images.to(p_device), l_labels.to(p_device)
        
        l_output = p_model.forward(l_images)
        l_test_loss += p_criterion(l_output, l_labels).item()

        l_ps = torch.exp(l_output)
        l_equality = (l_labels.data == l_ps.max(dim=1)[1])
        l_accuracy += l_equality.type(torch.FloatTensor).mean()
    
    return l_test_loss, l_accuracy

# Training function
def do_deep_learning(p_model, p_trainloader, p_validloader, p_epochs, p_print_every, p_criterion, p_optimizer, p_device):
    
    l_steps = 0

    p_model.to(p_device)
    print("Starting to train model")
    p_model.train()

    for l_e in range(p_epochs):
        l_running_loss = 0
        for l_ii, (l_inputs, l_labels) in enumerate(p_trainloader):
            l_steps += 1
        
            l_inputs, l_labels = l_inputs.to(p_device), l_labels.to(p_device)
        
            p_optimizer.zero_grad()
        
            # Forward and backward passes
            l_outputs = p_model.forward(l_inputs)
            l_loss = p_criterion(l_outputs, l_labels)
            l_loss.backward()
            p_optimizer.step()
        
            l_running_loss += l_loss.item()
        
            if l_steps % p_print_every == 0:
                # Make sure network is in eval mode for inference
                p_model.eval()
                
                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    l_test_loss, l_accuracy = validation(p_model, p_validloader, p_criterion, p_device)
                
                print("Epoch: {}/{}.. ".format(l_e+1, p_epochs),
                      "Training Loss: {:.3f}.. ".format(l_running_loss/p_print_every),
                      "Validation Loss: {:.3f}.. ".format(l_test_loss/len(p_validloader)),
                      "Validation Accuracy: {:.3f}.. ".format(l_accuracy/len(p_validloader)))
            
                l_running_loss = 0
                
                # Make sure training is back on
                p_model.train()

# Testing function
def check_accuracy(p_model, p_testloader, p_device):    
    l_correct = 0
    l_total = 0
    with torch.no_grad():
        for l_data in p_testloader:
            l_images, l_labels = l_data
            
            l_images, l_labels = l_images.to(p_device), l_labels.to(p_device)
            
            l_outputs = p_model.forward(l_images)
            _, l_predicted = torch.max(l_outputs.data, 1)
            l_total += l_labels.size(0)
            l_correct += (l_predicted == l_labels).sum().item()

    print('Accuracy of the network on {} test images: {:.2f}'.format( l_total, (100*l_correct/l_total) ))

#_______________________________________________________________________________________________________
########################################################################################################

# Model
model = make_model.make_custom_model(output_size, hidden_size, drop_p, arch)

# Optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr)

# Training
do_deep_learning(model, trainloader, validloader, epochs, print_every, criterion, optimizer, device)

# Testing
check_accuracy(model, testloader, device)

# Saving model as checkpoint
checkpoint = {'hidden_size': hidden_size,
              'output_size': output_size,
              'state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'class_to_idx': train_dataset.class_to_idx,
              'learning_rate': lr,
              'drop_p': drop_p,
              'epochs': epochs,
              'arch': arch
             }

# Save checkpoint
torch.save(checkpoint, (save_dir + save_name))
