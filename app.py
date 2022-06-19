import os
import os.path as op
import shutil
import pandas as pd
import numpy as np
from torchvision import datasets, models, transforms
import torchvision.transforms as T
from torch import nn
import torch
import PIL
from os import listdir
from os.path import isfile, join
from torch.utils.data import Dataset
import cv2
transformation = transforms.Compose([
  transforms.Resize((224,224)),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])                                   
])
new_model = torch.load("bell.pt", map_location=torch.device('cpu')) #โมเดล
new_model.eval()

import glob
from random import shuffle
import urllib.request
import streamlit as st

st.title("American sign language predict!!!")
def predict(img, model):
    def Get_img(img, transformation):
        if transformation : 
          img = transformation(img)
        return img
    imgs = Get_img(img, transformation = transformation)
    imgs = imgs[None, :, :, :]
    label_encode = {
        'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25, 'del': 26, 'nothing': 27, 'space': 28
    }
    label_decode = {value: key for key, value in label_encode.items()}
    prediction = model(imgs)
    # # Display the test image
    img = np.array(img)
    st.image(img)
    st.success(f'Predicted As :{label_decode[int(prediction.argmax())]}'+' with probability :{:.2f}%'.format(np.max(prediction.detach().numpy())*10))



##################################
# sidebar
##################################
# sidebar title
st.sidebar.write('### Enter image to classify')

# image source selection
option = st.sidebar.radio('', ['Use a validation image', 'Use your own image'])
valid_images = glob.glob('images/valid/*/*')
shuffle(valid_images)

if option == 'Use a validation image':
    st.sidebar.write('### Select a validation image')
    fname = st.sidebar.selectbox('',
                                 valid_images)

else:
    st.sidebar.write('### Select an image to upload')
    fname = st.sidebar.file_uploader('',
                                     type=['png', 'jpg', 'jpeg'],
                                     accept_multiple_files=False)
    if fname is None:
        fname = valid_images[0]
img = PIL.Image.open(fname)
# infer
predict(img, new_model)
