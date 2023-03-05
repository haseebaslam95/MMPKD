

import os
import torch
import torchvision.transforms as transforms
import sys


class ConfigBasic:
    def __init__(self):
        
        self.use_wandb = True
        self.device = torch.device('cuda')
        self.num_workers = 12
        
        
        self.dataset = 'affectnet'; assert self.dataset in {'raf-db','affectnet'}
        self.batch_size = 512
        self.test_batch_size = 64
        self.class_balanced_sampler = False
        self.weights = [3.8418008921655047, 2.1400215749730314, 11.298597745394556, 20.415259048970903, 45.10050172467858, 75.63791743360505, 11.56060606060606, 76.70693333333334]


        if self.dataset == 'raf-db':
            # self.train_transform = transforms.Compose([transforms.Resize((224, 224)),transforms.RandomHorizontalFlip(),
            #                                        transforms.RandomApply([transforms.RandomRotation(20),
            #                                        transforms.RandomCrop(224,padding=32)],p=0.2),
            #                                        transforms.ToTensor(),
            #                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
            #                                        transforms.RandomErasing(scale=(0.02,0.25))])
                                            
            self.train_transform = transforms.Compose([transforms.Resize((224, 224)),
                                                   transforms.ToTensor()])
                                    
        if self.dataset == 'affectnet':
            self.train_transform = transforms.Compose([transforms.Resize((224, 224)),transforms.RandomHorizontalFlip(),
                                                   transforms.RandomApply([transforms.RandomAffine(20, scale=(0.8, 1), translate=(0.2, 0.2)),], p=0.7),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                                                   transforms.RandomErasing(scale=(0.02,0.25))])

            self.test_transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

            self.test_train_plot_transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()])

        
        self.model_type = 'basic'; assert self.model_type in {'basic','dan','dacl'}
        if self.dataset == 'raf-db':
            self.output_shape = 7
        if self.dataset == 'affectnet':
            self.output_shape = 8
        self.backbone = 'res18_pre'; assert self.backbone in {'res18_pre','res50','res101','seresnet50','swin','enet'}
        self.dropout_prob = 0.8
        
        self.backbone_lr = 1e-4
        self.model_lr = 1e-4


class ConfigDAN:
    def __init__(self):
        
        self.use_wandb = False
        self.device = torch.device('cuda')
        self.num_workers = 12
        
        self.dataset = 'affectnet'; assert self.dataset in {'raf-db','affectnet'}
        self.batch_size = 256
        self.test_batch_size = 256
        self.class_balanced_sampler = True
        if self.dataset == 'raf-db':
            self.train_transform = transforms.Compose([transforms.Resize((224, 224)),transforms.RandomHorizontalFlip(),
                                                   transforms.RandomApply([transforms.RandomRotation(20),
                                                   transforms.RandomCrop(224,padding=32)],p=0.2),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                                                   transforms.RandomErasing(scale=(0.02,0.25))])
                                    
        if self.dataset == 'affectnet':
            self.train_transform = transforms.Compose([transforms.Resize((224, 224)),transforms.RandomHorizontalFlip(),
                                                   transforms.RandomApply([transforms.RandomAffine(20, scale=(0.8, 1), translate=(0.2, 0.2)),], p=0.7),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                                                   transforms.RandomErasing()])

        self.test_transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

        self.test_train_plot_transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()])

        
        self.model_type = 'dan'; assert self.model_type in {'basic','dan','dacl'}
        self.backbone = 'res18_pre'; assert self.backbone in {'res18_pre','res50','res101'}
        self.pretrained = True
        self.num_head = 4
        if self.dataset == 'raf-db':
            self.num_class = 7
        if self.dataset == 'affectnet':
            self.num_class = 8
        self.feat_dim = 512
        
        
        self.lr = 1e-4
        
        
        
        
        
        
        
        
class Path:
    def __init__(self,cfg,dataset_path,affnet_path=None):
        self.raf_db_path = os.path.join(dataset_path,'RAF-DB')
        self.raf_db_basic_path = os.path.join(self.raf_db_path,'basic')
        self.raf_db_compound_path = os.path.join(self.raf_db_path,'compound')

        self.raf_annotation_path = os.path.join(self.raf_db_basic_path,'Annotation')
        self.raf_images_path = os.path.join(os.path.join(self.raf_db_basic_path,'Image'),'original')
        self.raf_aligned_images_path = os.path.join(os.path.join(self.raf_db_basic_path,'Image'),'aligned')
        self.raf_emotion_file = os.path.join(os.path.join(self.raf_db_basic_path,'EmoLabel'),'list_patition_label.txt')
        self.landmarks_path = os.path.join(self.raf_annotation_path,'auto')

        
        self.backbone_model_path = os.path.join('Pre_trained_models','resnet18_msceleb.pth') # windows
        self.dan_saved_rafdb = os.path.join('Pre_trained_models','rafdb_epoch21_acc0.897_bacc0.8275.pth')
        self.dan_saved_affectnet8 = os.path.join('Pre_trained_models','affecnet8_epoch5_acc0.6209.pth')
        self.dan_saved_affectnet7 = os.path.join('Pre_trained_models','affecnet7_epoch6_acc0.6569.pth')
        
        self.affectnet_path = os.path.join(dataset_path,'AffectNet_Small')
        self.affectnet_train_path = os.path.join(self.affectnet_path,'train_set')
        self.affectnet_val_path = os.path.join(self.affectnet_path,'val_set')
        
        self.affectnet_train_images_path = os.path.join(self.affectnet_train_path,'images')
        self.affectnet_val_images_path = os.path.join(self.affectnet_val_path,'images')
        
        self.affectnet_train_anno_path = os.path.join(self.affectnet_train_path,'annotations')
        self.affectnet_val_anno_path = os.path.join(self.affectnet_val_path,'annotations')
