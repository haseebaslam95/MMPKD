from unittest.util import safe_repr
import cv2
import sys
import os
import csv

from src.detector import detect_faces
from utils.visualization_utils import show_bboxes
from PIL import Image
from facenet_pytorch import MTCNN
import  pandas as pd
from config1 import ConfigBasic
import torch.nn as nn
import torch
import csv
import ffmpeg
import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import pylab

file_name = '1_dev_1.wav' # sample file name
folder_path = '' #rootpath
audio_des_path = '' #path to extracted audio segments


counter=0
for audio_folder in os.listdir(audio_des_path):
    save_folder_path= folder_path+ '/'+ audio_folder+ '_spec'
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)
    
    for audio_file in os.listdir(audio_des_path+'/'+audio_folder):
        filesave=audio_file.split('.')
        print(audio_file)
        sig, fs = librosa.load(audio_des_path+'/'+audio_folder+'/'+audio_file)   
        # make pictures name 
        save_path = save_folder_path+ '/' +filesave[0] +'.jpg'
        

        pylab.axis('off') # no axis
        pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
        S = librosa.feature.melspectrogram(y=sig, sr=fs)
        librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
        pylab.savefig(save_path, bbox_inches=None, pad_inches=0)
        pylab.close()

        














