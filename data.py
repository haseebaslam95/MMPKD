from cgi import test
import random
import numpy as np
import os
import glob
import h5py
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
from tqdm import tqdm
import torchvision

from PIL import Image

from config import Path, ConfigCNNLSTM

class CustomDatasetS(Dataset):
    '''
    Custom Dataset Class for Training. Configurations are defined in config.py file.
    Details: Sample a random index in the video and extract consecutive frames of length clip length.
    Remarks: Looping might slow down training. Can be avoided by storing the tensors of images first in the momery.
    
    '''
    def __init__(self,anno_dict,feature_dict,mapping_dict,cfg,path,transform=None):
        self.anno_dict = anno_dict
        self.feature_dict = feature_dict
        self.mapping_dict = mapping_dict
        #self.clip_length = cfg.clip_length
        self.transform = transform
        self.path=path
        
    def __len__(self):
        return len(self.anno_dict)
    
    def __getitem__(self,idx):
        name = self.mapping_dict[idx]
        indx = random.randint(1,len(self.anno_dict[name])+1) 
        labels = self.anno_dict[name][indx]
        img_path = os.path.join(self.path.images_path,self.feature_dict[name][indx])
        print(img_path)
        img = self.transform(Image.open(img_path).convert('RGB'))
                  
        return img,labels
class CustomDatasetSTVAL(Dataset):
    '''
    Custom Dataset Class for Training. Configurations are defined in config.py file.
    Details: Sample a random index in the video and extract consecutive frames of length clip length.
    Remarks: Looping might slow down training. Can be avoided by storing the tensors of images first in the momery.
    
    '''
    def __init__(self,anno_dict,feature_dict,mapping_dict,cfg,path,transform=None):
        self.anno_dict = anno_dict
        self.feature_dict = feature_dict
        self.mapping_dict = mapping_dict
        self.clip_length = cfg.clip_length
        self.transform = transform
        self.path=path
        
    def __len__(self):
        return len(self.anno_dict)
    
    def __getitem__(self,idx):
        name = self.mapping_dict[idx]
        # indx = random.randint(1,len(self.anno_dict[name])-(self.clip_length+1)) 
        labels = []
        images = []
        vid_len=len(self.anno_dict[name])
        for i in range(1,vid_len+1):
            labels.append(self.anno_dict[name][i])
            img_path = self.feature_dict[name][i]
            img_path = os.path.join(self.path.images_path,img_path)
            img = self.transform(Image.open(img_path).convert('RGB'))
            images.append(img)
        images = torch.stack(images)
        labels = torch.stack(labels)      
        return images,labels,name


class CustomDatasetST(Dataset):
    '''
    Custom Dataset Class for Training. Configurations are defined in config.py file.
    Details: Sample a random index in the video and extract consecutive frames of length clip length.
    Remarks: Looping might slow down training. Can be avoided by storing the tensors of images first in the momery.
    
    '''
    def __init__(self,anno_dict,feature_dict,spec_dict,mapping_dict,cfg,path,transform=None):
        self.anno_dict = anno_dict
        self.spec_dict=spec_dict
        self.feature_dict = feature_dict
        self.mapping_dict = mapping_dict
        self.clip_length = cfg.clip_length
        self.transform = transform
        self.path=path
        
    def __len__(self):
        return len(self.anno_dict)
    
    def __getitem__(self,idx):
        name = self.mapping_dict[idx]
        indx = random.randint(1,len(self.anno_dict[name])-(self.clip_length+1)) 
        labels = []
        images = []
        specs = []
        for i in range(self.clip_length):
            normal_label=self.anno_dict[name][indx+i]
            labels.append(normal_label)
            # if normal_label==1:
            #     labels.append(torch.tensor([0,1]))
            # else:
            #     labels.append(torch.tensor([1,0]))
            
            img_path = self.feature_dict[name][indx+i]
            img_path = os.path.join(self.path.images_path,img_path)
            img = self.transform(Image.open(img_path).convert('RGB'))
            images.append(img)

            spec_path = self.spec_dict[name][indx+i]
            spec_path = os.path.join(self.path.specs_path,spec_path)
            spc = self.transform(Image.open(spec_path).convert('RGB'))
            specs.append(spc)

        images = torch.stack(images)
        labels = torch.stack(labels)
        specs = torch.stack(specs)      
        return images,specs,labels



class DataHandler:
    def __init__(self,path,cfg):
        
        self.cfg = cfg
        self.path = path
        
        self.train_csv_list = glob.glob(path.train_csv_path+os.sep+'*') 
        
        self.val_csv_list = glob.glob(path.val_csv_path+os.sep+'*')
        

        print('Preparing Training data dictionary')
        self.train_image_info, self.train_anno_dict, self.train_spec_info = self.load_data_info(self.train_csv_list,cfg)
        print('Preparing Validation data dictionary')
        self.val_image_info, self.val_anno_dict, self.val_spec_info = self.load_data_info(self.val_csv_list,cfg)

        self.train_mapping_dict = self.mapping_fn(self.train_anno_dict)
        self.val_mapping_dict = self.mapping_fn(self.val_anno_dict)

        self.train_transform = cfg.train_transform

        self.test_transform = cfg.test_transform

        self.val_list=list(self.val_anno_dict.keys())
        print(self.train_transform,)

    def getdataloader(self,cfg):
        train_dset = CustomDatasetST(self.train_anno_dict,self.train_image_info,self.train_spec_info,self.train_mapping_dict,
                                   cfg,self.path,transform=self.train_transform)
        train_loader = DataLoader(train_dset,batch_size = cfg.batch_size, num_workers = cfg.num_workers,
                                  shuffle = True,drop_last=True)
        val_dset = CustomDatasetST(self.val_anno_dict,self.val_image_info,self.val_spec_info,self.val_mapping_dict,
                                   cfg,self.path,transform=self.test_transform)

        val_loader = DataLoader(val_dset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=False,drop_last=True )
        self.train_dset=train_dset

        
        return train_loader, val_loader

    def collate_fn(batch):
        data = [item[0] for item in batch]
        images = torch.stack(data,0)
        
        
        label = [item[1] for item in batch]
        label = torch.stack(label,0)

        return images,label

    # def prepare_test_data(self,test_dset):
    #     with h5py.File('val_tensor112.hdf5', 'w') as f:
    #         for i in tqdm(range(len(test_dset))):
    #             features=test_dset[i][0]
    #             ky=test_dset[i][2]
    #             # image_tensor = []
    #             # features = feature_dict[val_ide]
    #             f.create_dataset(ky,data=features.numpy())
        
    def prepare_test_data(self,path,feature_dict,spec_dict,val_list,transform):
        print('Preparing Validation Data File')
        val_image_info= {}
        with h5py.File('val_tensor112.hdf5', 'w') as f:
            for val_ide in tqdm(val_list):
                image_tensor = []
                spec_tensor= []
                #spec tensor 
                features = feature_dict[val_ide]
                specs=spec_dict[val_ide]
                for i in range(1,len(features)+1):
                    # print(features[i])
                    imgpath=os.path.join(path.images_path,features[i])
                    spcpath=os.path.join(path.specs_path,specs[i])
                    img = transform(Image.open(imgpath).convert('RGB'))
                    image_tensor.append(img)
                    spc=transform(Image.open(spcpath).convert('RGB'))
                    spec_tensor.append(spc)
                image_tensor = torch.stack(image_tensor)
                spec_tensor =torch.stack(spec_tensor)
                #spec tensor as tuple
                f.create_dataset(val_ide,data=(image_tensor.numpy(),spec_tensor.numpy()))


  

    def load_data_info(self,csv_list,cfg):
        image_info_dict = {}
        anno_info_dict = {}
        spec_info_dict = {}
        label_column=cfg.mode


        for ide in tqdm(csv_list):
            df = pd.read_csv(ide)
            tmp_image_dict = {}
            tmp_anno_dict = {}
            tmp_spec_dict = {}

            
            images, labels, specs = df['image_name'], df[label_column], df['spec_name']
            idx = list(range(1,len(images)+1))
            for i in range(len(images)):
                tmp_image_dict[idx[i]] = images[i]
                tmp_spec_dict[idx[i]] = specs[i]
                tmp_anno_dict[idx[i]] = torch.tensor([labels[i]])
            image_info_dict[ide.split(os.sep)[-1][:-4]] = tmp_image_dict
            anno_info_dict[ide.split(os.sep)[-1][:-4]] = tmp_anno_dict
            spec_info_dict[ide.split(os.sep)[-1][:-4]] = tmp_spec_dict

            
        return image_info_dict, anno_info_dict, spec_info_dict

    def mapping_fn(self,dic):
        mapping_dict = {}
        i = 0
        for k,v in dic.items():
            mapping_dict[i] = k
            i += 1
        return mapping_dict


    