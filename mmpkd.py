from __future__ import annotations
from cProfile import label
from copy import copy
from modulefinder import Module
from pickletools import optimize
# from msilib.schema import Feature
import random
import re
from turtle import forward
from xml.sax.handler import all_properties
import numpy as np
import os
from glob import glob
import h5py
import pandas as pd
import cv2
import torch
from torch import classes, nn
import torch.nn.functional as F 
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader,Dataset
from tqdm import tqdm
import torchvision
import numpy as np
from PIL import Image
# from skimage import io
import matplotlib.pyplot as plt
from models_kd import *
from sklearn.metrics import f1_score, accuracy_score, average_precision_score, precision_recall_curve, confusion_matrix
import math
from data import CustomDatasetST, DataHandler
from config import ConfigCNNLSTM, Path
import logging
from CCC_loss import CCCLoss
from copy import deepcopy
import json
import pickle
import comet_ml
from thop import profile

experiment = comet_ml.Experiment(
    api_key="U0t7sSZhwEHvDLko0tJ4kbPH0",
    project_name="mmkd"
)



rootpath= '/home/livia/work/Recola_KD'
cfg=ConfigCNNLSTM()
path=Path(rootpath)


parameters = {'batch_size': cfg.batch_size,
              'clip_length':cfg.clip_length,
              'learning_rate_cnn': cfg.cnn_lr,
              'learning_rate_rnn': cfg.rnn_lr,
              'learning_rate_cnnspec': cfg.cnn_spec_lr,
              'epochs':cfg.num_epochs            
              }
experiment.log_parameters(parameters)


data_handler=DataHandler(path,cfg)

Logfile_name = "LogFiles/" + "log_file.log"
logging.basicConfig(filename=Logfile_name, level=logging.INFO)

train_loader, test_loader = data_handler.getdataloader(cfg)

#only once to prepare hdf5 file
# data_handler.prepare_test_data(path,data_handler.val_image_info,data_handler.val_spec_info,data_handler.val_list,data_handler.test_transform)



def unmapping_dict(a_dict):
    for k,v in a_dict.items():
        a_value=[]
        for kk,vv in v.items():
            a_value.append(vv)
        a_dict[k]=a_value
    return a_dict    




def calc_ccc(r_anno_dict,pred_dict,mode='train'):
    all_anno=[]
    all_predictions=[]
    total_ccc=0
    anno_dict = deepcopy(r_anno_dict)
    anno_dict=unmapping_dict(anno_dict)
    for k in anno_dict.keys():
        total_ccc+=check_ccc(torch.tensor(anno_dict[k]),torch.tensor(pred_dict[k]))
        all_anno.append(anno_dict[k])
        all_predictions.append(pred_dict[k])
    total_ccc/=len(anno_dict)
    return total_ccc




best_result=0


def check_ccc(x,y):
    # x = x.squeeze()
    y = y.squeeze(1)
    x_mean = torch.mean(x)
    y_mean = torch.mean(y)
    vx = x - x_mean
    vy = y - y_mean
    rho =  torch.sum(vx * vy) / (torch.sqrt(torch.sum(torch.pow(vx, 2))) * torch.sqrt(torch.sum(torch.pow(vy, 2))) + 1e-08)
    x_s = torch.std(x)
    y_s = torch.std(y)
    ccc = 2*rho*x_s*y_s/(torch.pow(x_s, 2) + torch.pow(y_s, 2) + torch.pow(x_mean - y_mean, 2) + 1e-08)

    return ccc





def predict(path,val_list,feature_dict,spec_dict,model_teacher,cfg,transform):
    model_teacher.cnn.eval()
    model_teacher.rnn.eval()
    model_teacher.spec_cnn.eval()

    val_pred_dict = {}
    with torch.no_grad():
        for val_ide in tqdm(val_list):
            image_tensor = []
            spec_tensor=[]
            features = feature_dict[val_ide] 
            specs = spec_dict[val_ide]
            for i in range(1,len(features)+1):

                img_path = features[i]
                img_path = os.path.join(path.images_path,img_path)
                img = Image.open(img_path).convert('RGB')
                image_tensor.append(transform(img))

                spc_path = specs[i]
                spc_path = os.path.join(path.specs_path,spc_path)
                spc = Image.open(spc_path).convert('RGB')
                spec_tensor.append(transform(spc))


            image_tensor = torch.stack(image_tensor)
            spec_tensor = torch.stack(spec_tensor)

            image_tensor = torch.split(image_tensor,split_size_or_sections=cfg.clip_length,dim=0)
            spec_tensor = torch.split(spec_tensor,split_size_or_sections=cfg.clip_length,dim=0)
            new_pred_dic = {}
            output_list = []
            i = 1
            h0 = torch.zeros(cfg.lstm_layer*cfg.lstm_num_dir,1,cfg.lstm_hidden_size).to(cfg.device)
            c0 = torch.zeros(cfg.lstm_layer*cfg.lstm_num_dir,1,cfg.lstm_hidden_size).to(cfg.device)
            for image,sp in zip(image_tensor,spec_tensor):
                final_out,h0,c0 = model_teacher.predict(image.to(cfg.device),h0,c0,sp.to(cfg.device))
                output_list.append(final_out.squeeze(1))
                c0 = c0[0]

            output_list = torch.cat(output_list,dim=0)
            val_pred_dict[val_ide] = output_list.cpu()

    return val_pred_dict



def predict_faster(model_teacher,cfg,val_file='val_tensor112.hdf5'):
    model_teacher.cnn.eval()
    model_teacher.rnn.eval()
    model_teacher.spec_cnn.eval()

    val_pred_dict = {}
    with torch.no_grad():
        f1=h5py.File(val_file,'r+')
        for key in tqdm(f1.keys()):
  
            image_tensor,spec_tensor = torch.tensor(f1[key][:])
            image_tensor = torch.split(image_tensor,split_size_or_sections=cfg.clip_length,dim=0)
            spec_tensor = torch.split(spec_tensor,split_size_or_sections=cfg.clip_length,dim=0)
            new_pred_dic = {}
            output_list = []
            i = 1
            h0 = torch.zeros(cfg.lstm_layer*cfg.lstm_num_dir,1,cfg.lstm_hidden_size).to(cfg.device)
            c0 = torch.zeros(cfg.lstm_layer*cfg.lstm_num_dir,1,cfg.lstm_hidden_size).to(cfg.device)
            for image,sp in zip(image_tensor,spec_tensor):
                final_out,h0,c0 = model_teacher.predict(image.to(cfg.device),h0,c0,sp.to(cfg.device))
                output_list.append(final_out.squeeze(1))
                c0 = c0[0]

            output_list = torch.cat(output_list,dim=0)
            val_pred_dict[key] = output_list.cpu()
        f1.close()
    return val_pred_dict

def predict_faster_student(model,cfg,val_file='val_tensor112.hdf5'):
    model.cnn.eval()
    model.rnn.eval()
    

    val_pred_dict = {}
    with torch.no_grad():
        f1=h5py.File(val_file,'r+')
        for key in tqdm(f1.keys()):
  
            image_tensor,spec_tensor = torch.tensor(f1[key][:])
            image_tensor = torch.split(image_tensor,split_size_or_sections=cfg.clip_length,dim=0)
            spec_tensor = torch.split(spec_tensor,split_size_or_sections=cfg.clip_length,dim=0)
            new_pred_dic = {}
            output_list = []
            i = 1
            h0 = torch.zeros(cfg.lstm_layer*cfg.lstm_num_dir,1,cfg.lstm_hidden_size).to(cfg.device)
            c0 = torch.zeros(cfg.lstm_layer*cfg.lstm_num_dir,1,cfg.lstm_hidden_size).to(cfg.device)
            for image,sp in zip(image_tensor,spec_tensor):
                final_out,h0,c0 = model.predict(image.to(cfg.device),h0,c0)
                output_list.append(final_out.squeeze(1))
                c0 = c0[0]

            output_list = torch.cat(output_list,dim=0)
            val_pred_dict[key] = output_list.cpu()
        f1.close()
    return val_pred_dict




# def train_model():
    


#     pass

# def train_loop():
#     for i in range (18):
#         train_model()
    

#     pass


#teacher model

model_teacher=CNN_LSTM_Teacher(cfg).to(cfg.device)

t_para = sum(p.numel() for p in model_teacher.parameters())

#student_model


# print('Loading State Dict')
# model.load_state_dict(torch.load(path.model_save_path))

# print(model_teacher)



# criterion = nn.MSELoss()
criterion = CCCLoss().to(cfg.device)
# criterion = nn.KLDivLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)



# check=iter(train_loader)
# image,labels=check.next()
# print(image.shape)
# data_handler.prepare_test_data(test_dset)

val_list=list(data_handler.val_anno_dict.keys())
train_list=list(data_handler.train_anno_dict.keys())
n_total_steps = len(train_loader)
val_anno_dict=data_handler.val_anno_dict




##### Teacher Training ###########

def train_teacher(cfg):
    best_epoch=0
    best_ccc=-2.0
    print('***Starting Teacher Training...***')
    for epoch in tqdm(range(cfg.num_epochs)):

        for i, (features,specs,classes) in enumerate(train_loader):
            model_teacher.cnn.train()
            model_teacher.rnn.train()
            model_teacher.spec_cnn.train()
            specs=specs.to(cfg.device)
            features = features.to(cfg.device)
            classes = classes.to(cfg.device)
            intermediate_out_teacher,outputs = model_teacher.model_out(features,specs)
            loss = criterion(outputs.view(-1,1).float(),classes.view(-1,1).float())
            
            model_teacher.cnn_optimizer.zero_grad()
            model_teacher.rnn_optimizer.zero_grad()
            model_teacher.spec_cnn_optimizer.zero_grad()
            
            loss.backward()
            model_teacher.spec_cnn_optimizer.step()
            model_teacher.cnn_optimizer.step()
            model_teacher.rnn_optimizer.step()



        if(epoch%cfg.watch_every==0 and epoch > 1):



            print(f'Epoch[{epoch+1}/{cfg.num_epochs}], Loss: {loss:.4f}')

            #Check predictions with Train data
            
            
            # train_pred_dict=predict(path,train_list,data_handler.train_image_info,data_handler.train_spec_info,model,cfg,transform=data_handler.train_transform)
            # ccc= calc_ccc(data_handler.train_anno_dict,train_pred_dict,mode='eval')


            # Check predictions with Val Data

            # val_pred_dict=predict(path,val_list,data_handler.val_image_info,data_handler.val_spec_info,model,cfg,transform=data_handler.test_transform)
            val_pred_dict=predict_faster(model_teacher,cfg)   #check faster with val_tensor
            ccc= calc_ccc(val_anno_dict,val_pred_dict,mode='eval')
            print(ccc)
            metrics = {'CCC': ccc, 'loss': loss}
            experiment.log_metrics(metrics)



            if ccc>best_ccc:
                best_ccc=ccc
                best_epoch=epoch
                if cfg.mode=='arousal':
                    torch.save(model_teacher.state_dict(), path.model_save_path_teacher_arousal)
                elif cfg.mode=='valence':
                    torch.save(model_teacher.state_dict(), path.model_save_path_teacher_valence)

                print('Best_Epoch: ', best_epoch)
                print('Best_CCC: ', best_ccc)

                

            logging.info('At Epoch')
            logging.info(epoch)
            logging.info('CCC')
            logging.info(ccc)

    return best_ccc, best_epoch

                


### ---------- Visual only Training -------####

v_model=CNN_LSTM_s(cfg).to(cfg.device)

def train_visual(cfg):
    best_epoch=0
    best_ccc=-2.0
    print('***Starting Training for visual only...***')
    for epoch in tqdm(range(cfg.num_epochs)):

        for i, (features,specs,classes) in enumerate(train_loader):
            v_model.cnn.train()
            v_model.rnn.train()
            features = features.to(cfg.device)
            classes = classes.to(cfg.device)
            intermediate_out_student,outputs = v_model.model_out(features)
            loss = criterion(outputs.view(-1,1).float(),classes.view(-1,1).float())
            v_model.cnn_optimizer.zero_grad()
            v_model.rnn_optimizer.zero_grad()
            loss.backward()
            v_model.cnn_optimizer.step()
            v_model.rnn_optimizer.step()
            # print(intermediate_out_student.shape)


        if(epoch%cfg.watch_every==0):
            print(f'Epoch[{epoch+1}/{cfg.num_epochs}], Loss: {loss:.4f}')
            
            # val_pred_dict=predict(path,val_list,data_handler.val_image_info,model,cfg,transform=data_handler.test_transform)
            val_pred_dict=predict_faster_student(v_model,cfg)
            ccc= calc_ccc(val_anno_dict,val_pred_dict,mode='eval')
            # val_anno_dict=data_handler.val_anno_dict
            # val_pred_dict=predict(path,val_list,data_handler.val_image_info,model,cfg,transform=data_handler.test_transform)
            # ccc= calc_ccc(val_anno_dict,val_pred_dict,mode='eval')
            print(ccc)
            if ccc>best_ccc:
                best_ccc=ccc
                best_epoch=epoch
                torch.save(v_model.state_dict(), path.model_save_path_v)
                print('Best_Epoch: ', best_epoch)
                print('Best_CCC: ', best_ccc)

    return best_ccc, best_epoch



### ---------- Audio only Training -------####







### Student Training ####

model_student=CNN_LSTM_s(cfg).to(cfg.device)
s_para = sum(p.numel() for p in model_student.parameters())


print(t_para)
print(s_para)

print(t_para-s_para)

def train_student(cfg):
    ###teacher model load
    if cfg.mode=='arousal':
        model_load_path_teacher=path.model_save_path_teacher_arousal
    elif cfg.mode=='valence':
        model_load_path_teacher=path.model_save_path_teacher_valence
    
    model_teacher.load_state_dict(torch.load(model_load_path_teacher))
    cos_similarity = nn.CosineSimilarity(dim=0, eps=1e-6)
    alpha = 0.8
    beta = 1 - alpha
    gamma=0.2
    theta=cfg.theta
    best_epoch=0
    best_ccc=-2.0
    with experiment.train():
        for epoch in tqdm(range(cfg.num_epochs)):

            ### get teacher intermediate embeddings

            for i, (features,specs,classes) in enumerate(train_loader):
                # model_teacher.cnn.train()
                # model_teacher.rnn.train()
                # model_teacher.spec_cnn.train()


                specs=specs.to(cfg.device)
                features = features.to(cfg.device)
                classes = classes.to(cfg.device)

                intermediate_output_teacher, t_outputs = model_teacher.model_out(features,specs)
                gamma = criterion(t_outputs.view(-1,1).float().detach(),classes.view(-1,1).float())

                ######## gamma calc ###########
                if gamma <= theta or gamma > 1:
                    beta = 0.2 + gamma
                elif gamma > theta:
                    beta = 1

            ########### ******** Student training **********

                

                model_student.cnn.train()
                model_student.rnn.train()
                features = features.to(cfg.device)
                classes = classes.to(cfg.device)
                intermediate_output_student , outputs = model_student.model_out(features)

                cosloss= 1 - cos_similarity(intermediate_output_teacher.view(-1,1).detach(),intermediate_output_student.view(-1,1))

                ccc_loss = criterion(outputs.view(-1,1).float(),classes.view(-1,1).float()) 
                
                ##### neg transfer ####

                alpha=1-beta

                loss = (alpha * ccc_loss) + (beta* cosloss)
                
                # loss=ccc_loss

                model_student.cnn_optimizer.zero_grad()
                model_student.rnn_optimizer.zero_grad()

                loss.backward() 

                model_student.cnn_optimizer.step()
                model_student.rnn_optimizer.step()




            if(epoch%cfg.watch_every==0 and epoch > 1):
                print(f'Epoch[{epoch+1}/{cfg.num_epochs}], Loss: {loss.item():.4f}')
                # train_pred_dict=predict(path,train_list,data_handler.train_image_info,data_handler.train_spec_info,model,cfg,transform=data_handler.train_transform)
                # ccc= calc_ccc(data_handler.train_anno_dict,train_pred_dict,mode='eval')

                # val_pred_dict=predict(path,val_list,data_handler.val_image_info,data_handler.val_spec_info,model,cfg,transform=data_handler.test_transform)
                val_pred_dict=predict_faster_student(model_student,cfg)
                ccc= calc_ccc(val_anno_dict,val_pred_dict,mode='eval')
                # val_anno_dict=data_handler.val_anno_dict
                # val_pred_dict=predict(path,val_list,data_handler.val_image_info,model,cfg,transform=data_handler.test_transform)
                # ccc= calc_ccc(val_anno_dict,val_pred_dict,mode='eval')
                
                experiment.log_metric('Loss', loss,epoch= epoch)
                experiment.log_metric('CCC', ccc,epoch= epoch)
                print(ccc)
                if ccc>best_ccc:
                    best_ccc=ccc
                    best_epoch=epoch
                    if cfg.mode=='arousal':
                        torch.save(model_student.state_dict(), path.model_save_path_student_arousal)
                    elif cfg.mode=='valence':
                        torch.save(model_student.state_dict(), path.model_save_path_student_valence)

                    print('Best_Epoch: ', best_epoch)
                    print('Best_CCC: ', best_ccc)

                    
                # if f1>best_f1:
                #     best_f1=f1
                #     best_epoch=epoch
                #     torch.save(model.state_dict(), path.model_save_path)

                # print('At Epoch: ', epoch)    
                # print('Train Accuracy:',acc)
                # print('Train F1', f1) 
                # print('Train CF:\n',cf)
                logging.info('At Epoch')
                logging.info(epoch)
                logging.info('CCC')
                logging.info(ccc)
                # logging.info('Training F1 Score:')
                # logging.info(f1)
                # logging.info('Training Confusion Matrix:')
                # logging.info(cf)
            # if(epoch%cfg.watch_every==0):
            #     check_val(cfg,data_handler)
        return best_ccc, best_epoch













###Teacher training call ####

# best_ccc, best_epoch = train_teacher(cfg)
# print('Finished Training')
# print('Best_Epoch: ', best_epoch)
# print('Best_CCC: ', best_ccc)

# logging.info('Finished Training')
# logging.info('Best Epoch')
# logging.info(best_epoch)
# logging.info('Best CCC')
# logging.info(best_ccc)

# print('***Loading best model...***')
# val_pred_dict=predict_faster(model_teacher,cfg)
# val_ccc= calc_ccc(val_anno_dict,val_pred_dict,mode='eval')

# print('Val CCC: ', val_ccc)

# logging.info('Val_CCC')
# logging.info(val_ccc)

# print('***Loading best model...***')
# model_load_path=path.model_save_path_teacher_arousal
# model_teacher.load_state_dict(torch.load(model_load_path))

# val_pred_dict=predict_faster(model_teacher,cfg)


###*******visual training call************
# best_ccc, best_epoch=train_visual(cfg)
# print('***Finished Training for visual only model...***')
# print('Best_Epoch: ', best_epoch)
# print('Best_CCC: ', best_ccc)


## *******Student training call*************

best_ccc, best_epoch=train_student(cfg)
print('***Finished Training for Student model...***')
print('Best_Epoch: ', best_epoch)
print('Best_CCC: ', best_ccc)


print('***Loading best model...***')
model_load_path=path.model_save_path_student_arousal
model_student.load_state_dict(torch.load(model_load_path))


# model_student.load_state_dict(torch.load(path.model_save_path_student))
# val_pred_dict=predict(path,val_list,data_handler.val_image_info,data_handler.val_spec_info,model,cfg,transform=data_handler.test_transform)

val_pred_dict=predict_faster_student(model_student,cfg)
val_ccc= calc_ccc(val_anno_dict,val_pred_dict,mode='eval')

# print('Val CCC: ', val_ccc)

# logging.info('Val_CCC')
# logging.info(val_ccc)




def save_preds(r_anno_dict,pred_dict):
    all_anno=[]
    all_predictions=[]
    # total_ccc=0
    anno_dict = deepcopy(r_anno_dict)
    anno_dict=unmapping_dict(anno_dict)
    for k in anno_dict.keys():
        all_anno.append(anno_dict[k])
        all_predictions.append(pred_dict[k])
    return all_anno,all_predictions




print('saving predictions and annotations')
if cfg.mode=='valence':
    with open('val_anno_dict_valence.pkl', 'wb') as f:
        pickle.dump(val_anno_dict, f)
    with open('val_pred_dict_valence.pkl', 'wb') as f:
        pickle.dump(val_pred_dict, f)

elif cfg.mode=='arousal':
    with open('val_anno_dict_arousal.pkl', 'wb') as f:
        pickle.dump(val_anno_dict, f)
    with open('val_pred_dict_arousal.pkl', 'wb') as f:
        pickle.dump(val_pred_dict, f)



