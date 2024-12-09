import os 
from glob import glob
import numpy as np 

import cv2 
import torch 
from torchvision import transforms



from facialKeyPointModel import get_model 

### 
WEIGHT = 'weights/facial_keypoints_detection_model.pth'

## load weights 
model = get_model(weight=WEIGHT)
model.eval()

cap = cv2.VideoCapture(0)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

while True:
    ret, frame = cap.read()
    if not ret:
        break 

    img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) / 255
    img = cv2.resize(img, (224,224))
    img = torch.tensor(img).permute(2,0,1)
    img = normalize(img).float()

    with torch.no_grad():
        kps = model(img[None]).flatten().detach().cpu()
    x, y= kps[:68], kps[68:]
    for i in range(len(x)):
        x_i, y_i = int(x[i]*frame.shape[1]), int(y[i]*frame.shape[0])
        cv2.circle(frame,(x_i,y_i), radius=5,color=(0,255,0), thickness=-1)
    cv2.imshow('Facial Keypoints', frame)

    if cv2.waitKey(1) & 0xFF== ord('q'): #press q  to quit 
        break 

cap.release()
cv2.destroyAllWindows() 


