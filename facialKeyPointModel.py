import torch 
from torch import nn 

import torchvision 
from torchvision import models 



def get_model(weight):
  model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
  for param in model.parameters():
        param.requires_grad = False
  model.avgpool = nn.Sequential( nn.Conv2d(512,512,3),
                                nn.BatchNorm2d(512),
                                nn.ReLU(),
                                nn.MaxPool2d(2),
                                nn.Flatten())
  model.classifier = nn.Sequential(
    nn.Linear(2048, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 136),
    nn.Sigmoid()
  )

  model = model.to('cpu')

  if weight:
      try:
          state_dict = torch.load(weight,map_location='cpu')
          model.load_state_dict(state_dict)
          print('Trained Weights Loaded')
      except FileNotFoundError:
          print('Wrong File')

  return model

