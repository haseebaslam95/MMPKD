import cv2
import sys
import os
import csv


from PIL import Image
from facenet_pytorch import MTCNN
import  pandas as pd
from basic import BasicModel
from config1 import ConfigBasic
import torch.nn as nn
import torch
import csv
import ffmpeg
import librosa
import soundfile as sf

file_name = 'dev_1.wav' #sample file name
folder_path = '' # path to audio raw audio files 
audio_des_path = '' # path to destination folder to save extracted audio segments

if not os.path.exists(audio_des_path):
    os.makedirs(audio_des_path)

for file in os.listdir(folder_path):

    # First load the file
    audio, sr = librosa.load(folder_path + '/' + file)


    # Get number of samples for 2 seconds; replace 2 by any number
    buffer = 1 * sr

    samples_total = len(audio)
    samples_wrote = 0
    counter = 1

    audio_file_split = file.split('.')

    audio_files_folder = audio_des_path + '/' + audio_file_split[0]

    if not os.path.exists(audio_files_folder):
        os.makedirs(audio_files_folder)

    while samples_wrote < samples_total:

        #check if the buffer is not exceeding total samples 
        if buffer > (samples_total - samples_wrote):
            buffer = samples_total - samples_wrote

        block = audio[samples_wrote : (samples_wrote + buffer)]
        out_filename = audio_files_folder + '/' + str(counter) + "_" + file

        # Write 2 second segment
        sf.write(out_filename, block, sr)    
        counter += 1
        samples_wrote += buffer
