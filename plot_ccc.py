import cv2
import sys
import os
import csv

from src.detector import detect_faces
from utils.visualization_utils import show_bboxes
from PIL import Image
from facenet_pytorch import MTCNN
import  pandas as pd
from config import *
from config1 import ConfigBasic
import torch.nn as nn
import torch
import csv
import itertools
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from collections import deque
import numpy as np

# mode='valence'
cfg=ConfigCNNLSTM()
mode=cfg.mode

vid_keys=['dev_1','dev_2','dev_3','dev_4','dev_5','dev_6','dev_7','dev_8','dev_9']

anno_dict_name='val_anno_dict.pkl'
pred_dict_name='val_pred_dict.pkl'

shift_factor=60
scale_factor=1.45
mean_factor=60

print('Shift Factor: ',shift_factor)
print('Scale Factor: ',scale_factor)
print('Mean Factor: ',mean_factor)

def check_ccc(x,y):
    # x = x.squeeze()
    # y = y.squeeze(1)
    x_mean = torch.mean(x)
    y_mean = torch.mean(y)
    vx = x - x_mean
    vy = y - y_mean
    rho =  torch.sum(vx * vy) / (torch.sqrt(torch.sum(torch.pow(vx, 2))) * torch.sqrt(torch.sum(torch.pow(vy, 2))) + 1e-08)
    x_s = torch.std(x)
    y_s = torch.std(y)
    ccc = 2*rho*x_s*y_s/(torch.pow(x_s, 2) + torch.pow(y_s, 2) + torch.pow(x_mean - y_mean, 2) + 1e-08)

    return ccc



def unmapping_dict(a_dict):
    for k,v in a_dict.items():
        a_value=[]
        for kk,vv in v.items():
            a_value.append(vv)
        a_dict[k]=a_value
    return a_dict    


def mean_fitler(factor,pred_list):
    k=factor
    kern=np.ones(2*k+1)/(2*k+1)
    out=np.convolve(pred_list,kern, mode='same')
    return out
    
    
def time_shift(factor, pred_list):
    pred_list= deque(pred_list)
    pred_list.rotate(factor)
    return pred_list
    
def scale(factor, pred_list):
    pred_list=[(x*factor) for x in pred_list]
    return pred_list



def calc_total_ccc(vid_key):
    
    with open(anno_dict_name, 'rb') as f:
        val_anno_dict = pickle.load(f)

    with open(pred_dict_name, 'rb') as f:
        val_pred_dict = pickle.load(f)

    val_anno_dict=unmapping_dict(val_anno_dict)

    pred_list=val_pred_dict[vid_key].squeeze(1).tolist()

    anno_list= [x.item() for x in val_anno_dict[vid_key]]

    scaled=scale(scale_factor,pred_list)
    shifted=time_shift(shift_factor,scaled)
    filtered= mean_fitler(mean_factor,shifted)
    ccc=check_ccc(torch.FloatTensor(anno_list),torch.FloatTensor(filtered))
    print(vid_key,ccc)
    return ccc


def generate_plots(vid_key):
    with open(anno_dict_name, 'rb') as f:
        val_anno_dict = pickle.load(f)

    with open(pred_dict_name, 'rb') as f:
        val_pred_dict = pickle.load(f)

    val_anno_dict=unmapping_dict(val_anno_dict)

    pred_list=val_pred_dict[vid_key].squeeze(1).tolist()   # for normal dicts
    #pred_list=val_pred_dict[vid_key].tolist()           # for audio only dicts

    anno_list= [x.item() for x in val_anno_dict[vid_key]]

    # pred_list=[(x*1.9) for x in pred_list]

    # rotated= deque(pred_list)
    # rotated.rotate(60)
    scaled=scale(scale_factor,pred_list)
    shifted=time_shift(shift_factor,scaled)
    filtered= mean_fitler(mean_factor,shifted)
    
    ccc=check_ccc(torch.FloatTensor(anno_list),torch.FloatTensor(filtered))
    print(vid_key,ccc)    
    
    x_axis= [i for i in range(7500)]

    if mode=='valence':
        plot_text=vid_key +' Valence CCC: '+str(np.round(ccc.item(),4))

    elif mode=='arousal':
        plot_text=vid_key +' Arousal CCC: '+str(np.round(ccc.item(),4))



    plt.figure(figsize=(100,10))

    ax = plt.gca()
    ax.set_xlim([0, 7500])
    ax.set_ylim([-1, 1])

    plt.plot(x_axis,filtered, label='Predictions')
    plt.plot(x_axis,anno_list, label='Annotations')
    plt.text(1,1,plot_text,size=50)
    # plt.plot(x_list, train_Z, label='train Z')
    # plt.legend()

    if mode=='valence':
        save_name='Valence_filtered' + str(vid_key) + '.png'
    elif mode=='arousal':
        save_name='Arousal_filtered' + str(vid_key) + '.png'
    plt.savefig(save_name)

    return ccc


tot_ccc=0

for vid_key in vid_keys:
    tot_ccc+=generate_plots(vid_key)


print(tot_ccc/9)











































# vid_keys=['dev_1','dev_2','dev_3','dev_4','dev_5','dev_6','dev_7','dev_8','dev_9']

# specs_folder_path='/home/livia/work/RecolaAV/specs'

# def check_ccc(x,y):
#     # x = x.squeeze()
#     # y = y.squeeze(1)
#     x_mean = torch.mean(x)
#     y_mean = torch.mean(y)
#     vx = x - x_mean
#     vy = y - y_mean
#     rho =  torch.sum(vx * vy) / (torch.sqrt(torch.sum(torch.pow(vx, 2))) * torch.sqrt(torch.sum(torch.pow(vy, 2))) + 1e-08)
#     x_s = torch.std(x)
#     y_s = torch.std(y)
#     ccc = 2*rho*x_s*y_s/(torch.pow(x_s, 2) + torch.pow(y_s, 2) + torch.pow(x_mean - y_mean, 2) + 1e-08)

#     return ccc



# def unmapping_dict(a_dict):
#     for k,v in a_dict.items():
#         a_value=[]
#         for kk,vv in v.items():
#             a_value.append(vv)
#         a_dict[k]=a_value
#     return a_dict    


# def calc_total_ccc(vid_key):
    
#     with open('val_anno_dict.pkl', 'rb') as f:
#         val_anno_dict = pickle.load(f)

#     with open('val_pred_dict_41.pkl', 'rb') as f:
#         val_pred_dict = pickle.load(f)

#     val_anno_dict=unmapping_dict(val_anno_dict)

#     pred_list=val_pred_dict[vid_key].squeeze(1).tolist()

#     anno_list= [x.item() for x in val_anno_dict[vid_key]]

#     pred_list=[(x*1.5) for x in pred_list]

#     rotated= deque(pred_list)
#     rotated.rotate(60)
#     ccc=check_ccc(torch.FloatTensor(anno_list),torch.FloatTensor(rotated))
#     print(vid_key,ccc)
#     return ccc


# def mean_fitler(pred_list):
#     k=60
#     kern=np.ones(2*k+1)/(2*k+1)
#     out=np.convolve(pred_list,kern, mode='same')
#     return out
    
    

    



# def generate_plots(vid_key):
#     with open('val_anno_dict.pkl', 'rb') as f:
#         val_anno_dict = pickle.load(f)

#     with open('val_pred_dict_arousal_22.pkl', 'rb') as f:
#         val_pred_dict = pickle.load(f)

#     val_anno_dict=unmapping_dict(val_anno_dict)

#     pred_list=val_pred_dict[vid_key].squeeze(1).tolist()

#     anno_list= [x.item() for x in val_anno_dict[vid_key]]

#     pred_list=[(x*1.9) for x in pred_list]

#     rotated= deque(pred_list)
#     rotated.rotate(60)
    
#     filtered= mean_fitler(rotated)
#     ccc=check_ccc(torch.FloatTensor(anno_list),torch.FloatTensor(filtered))
#     print(vid_key,ccc)    
    
#     x_axis= [i for i in range(7500)]

    


#     plt.figure(figsize=(100,10))

#     ax = plt.gca()
#     ax.set_xlim([0, 7500])
#     ax.set_ylim([-1, 1])

#     plt.plot(x_axis,filtered, label='Predictions')
#     plt.plot(x_axis,anno_list, label='Annotations')
#     # plt.plot(x_list, train_Z, label='train Z')
#     # plt.legend()
#     save_name='Arousal_filtered' + str(vid_key) + '.png'
#     plt.savefig(save_name)

#     return ccc


# tot_ccc=0

# for vid_key in vid_keys:
#     # tot_ccc += calc_total_ccc(vid_key)
#     tot_ccc+=generate_plots(vid_key)


# print(tot_ccc/9)


# print(pred_list)
# rootpath= '/home/livia/work/Ambivalence'

# all_videos_name= os.listdir(os.path.join(rootpath, "1.1"))
# # print(all_videos_name)

# df = pd.read_excel('annotations.xlsx',header=None)
# #print(df.head())
# names = df[0]
# strt = 0
# endt = 0
# for i in range(1,len(names)):
# 	if str(df[1][i]) == '1.1' and str(df[2][i])== '1.0':
# 		print(df[4][i])
# cfg1=ConfigBasic()
# model=BasicModel(cfg1)

# model_path = os.path.join('Pre_trained_models','affectNetRes18_small.pt')
# checkpoint = torch.load(model_path,map_location=cfg1.device)
# model.load_state_dict(checkpoint,strict=True)
# model = nn.Sequential(*list(model.children())[:-2])

# print(model)


