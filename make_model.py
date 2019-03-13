from torchvision import models
from torch import nn

from collections import OrderedDict

# Function to create new model from pre-defined model
def make_custom_model(p_output_size = 102, p_hidden_size = [], p_drop_p = 0.2, p_arch = 'vgg16'):
    
    if(p_arch == 'vgg16'):
        l_model = models.vgg16(pretrained=True)
        l_input_size  = 25088
        if len(p_hidden_size) == 0:
            p_hidden_size = [5000, 500]
    elif(p_arch == 'resnet18'):
        l_model = models.resnet18(pretrained=True)
        l_input_size  = 512
        if len(p_hidden_size) == 0:
            p_hidden_size = [325, 180]
    else:
        print('Only "vgg16" and "resnet18" models are supported')
        exit(1)
    
    # Freeze parameters
    for l_param in l_model.parameters():
        l_param.requires_grad = False

    l_model.classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(l_input_size, p_hidden_size[0])),
                            ('relu1', nn.ReLU()),
                            ('drop1', nn.Dropout(p = p_drop_p)),
                            ('fc2', nn.Linear(p_hidden_size[0], p_hidden_size[1])),
                            ('relu2', nn.ReLU()),
                            ('drop2', nn.Dropout(p = p_drop_p)),
                            ('fc3', nn.Linear(p_hidden_size[1], p_output_size)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))

    return l_model