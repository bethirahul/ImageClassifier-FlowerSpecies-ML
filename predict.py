#!/usr/bin/env python3

import numpy as np

import torch

import make_model

from PIL import Image

import json

import argparse

parser = argparse.ArgumentParser(add_help=True,
    description='This is a flower species prediction program',
)

parser.add_argument('image_file', action="store", default="flowers/test/89/image_00745.jpg",
                    help="Image file path")
parser.add_argument('--checkpoint', action="store", dest="checkpoint_file", default="checkpoint.pth",
                    help="Model Checkpoint file path")
parser.add_argument('--top_K', action="store", dest="top_K", default=3, type=int,
                    help="Top K categories in prediction result")
parser.add_argument('--category_names', action="store", dest="cat_names_file", default="cat_to_name.json",
                    help="Category Names to classify")
parser.add_argument('--gpu', action="store_true", dest="useOnlyGPU", default=False,
                    help="Should the model untilize only GPU?")

inputs = parser.parse_args()
image_file = inputs.image_file
checkpoint_file = inputs.checkpoint_file
top_K = inputs.top_K
cat_names_file = inputs.cat_names_file
useOnlyGPU = inputs.useOnlyGPU

print(inputs)

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

with open(cat_names_file, 'r') as f:
    cat_to_name = json.load(f)
    
#_______________________________________________________________________________________________________
########################################################################################################

# Loads a checkpoint and rebuilds the model
def load_model_checkpoint(p_checkpoint_file, p_device):
    l_checkpoint = torch.load(p_checkpoint_file)
    l_model = make_model.make_custom_model(l_checkpoint['output_size'],
                                           l_checkpoint['hidden_size'],
                                           l_checkpoint['drop_p'],
                                           l_checkpoint['arch'])
    l_model.load_state_dict(l_checkpoint['state_dict'])
    l_model.class_to_idx = l_checkpoint['class_to_idx']
    
    l_model.to(p_device)

    return l_model

def process_image(p_image_path, p_min_size, p_crop_size, p_mean_arr, p_std_arr):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    l_image = Image.open(p_image_path)
    l_width, l_height = l_image.size
    
    p_min_size = int(p_min_size)
    if l_width < l_height:
        l_image.resize((p_min_size, int(l_height * (p_min_size/l_width))))
    elif l_height > l_width:
        l_image.resize((int(l_width * (p_min_size/l_height)), p_min_size))
    
    # clockwise ||---|------||------|---||
    l_left   = (l_width  - p_crop_size)/2
    l_top    = (l_height - p_crop_size)/2
    l_right  = (l_width  + p_crop_size)/2
    l_bottom = (l_height + p_crop_size)/2
    l_image  = l_image.crop((l_left, l_top, l_right, l_bottom))
    
    # converts 0-255 color values to 0-1 floats
    l_image = np.array(l_image)/255
    
    p_mean_arr = np.array(p_mean_arr)
    p_std_arr  = np.array(p_std_arr)
    l_image    = (l_image - p_mean_arr)/p_std_arr
    
    # Move the color channel to first place [] [0] [1] [2]
    l_image = l_image.transpose((2, 0, 1)) # ^----------/
    
    return l_image

def predict(p_image_path, p_model, p_cat_to_name, p_device, p_top_K=3):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    l_min_size = 256
    l_crop_size = 224
    l_mean_arr = [0.485, 0.456, 0.406]
    l_std_arr = [0.229, 0.224, 0.225]
    
    l_processed_image = process_image(p_image_path, l_min_size, l_crop_size, l_mean_arr, l_std_arr)
    # Convert to Tensor
    l_image_tensor = l_processed_image.copy()
    l_image_tensor = torch.from_numpy(l_image_tensor).type(torch.FloatTensor)
    # Model and image must be on same device
    l_image_tensor = l_image_tensor.to(p_device)
    l_image_tensor = l_image_tensor.unsqueeze_(0)
    with torch.no_grad():
        p_model.eval()
        l_model_output = p_model(l_image_tensor)
    l_ps = torch.exp(l_model_output)
    l_probs, l_classes = l_ps.topk(p_top_K)
    
    l_idx_to_class = { l_value: l_key for l_key, l_value in p_model.class_to_idx.items() }
    
    l_probabilities = l_probs.cpu().data.numpy()[0]
    l_class_indices = l_classes.cpu().data.numpy()[0].tolist()
    l_class_names_indices = [ l_idx_to_class[l_x] for l_x in l_class_indices ]
    l_class_names = [ p_cat_to_name[str(l_x1)] for l_x1 in l_class_names_indices ]
    print(l_probabilities, '\n', l_class_names_indices, '\n', l_class_names)
    return dict(zip(l_class_names, l_probabilities))
    
#_______________________________________________________________________________________________________
########################################################################################################
    
model = load_model_checkpoint(checkpoint_file, device)

results = predict(image_file, model, cat_to_name, device, top_K)
print('Here are the top prediction results with category and its probability of prediction:')
for className, prob in results.items():
    print(f"{className}: {prob}")
