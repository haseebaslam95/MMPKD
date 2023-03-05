import os
import sys
import torch
import torchvision.transforms as transforms



class ConfigCNNLSTM:
    '''
    Hyperparameter settings for the CNN-RNN model.
    '''
    def __init__(self):
        #self.mode='valence'
        self.mode='arousal'
        self.use_wandb = False
        #device to run
        self.device = torch.device('cuda')
        
        
        #settings related to data batching and loading
        self.clip_length = 5
        self.batch_size = 8
        self.num_workers = 16
        self.train_transform = transforms.Compose([transforms.Resize((112,112)),
                                                   transforms.ToTensor()])

        self.test_transform = transforms.Compose([transforms.Resize((112,112)),transforms.ToTensor()])

        #settings related to model
        self.model_type = 'spatio-temporal'; assert self.model_type in {'spatial','spatio-temporal'}
        
        #settings related to backone CNN
        self.cnn_backbone_type = 'res18_pre'; assert self.cnn_backbone_type in {'res18','res18_pre','res50','res101','vit','mobilefacenet','dense121','dense169','swift'}
        self.cnn_bottleneck_size = 512  #Note: addition of bottleneck layer deteriotes the performance.
        
        
        #settings related to RNN
        self.rnn_type = 'gru'  #{'lstm','gru'}
        self.lstm_layer =1
        self.lstm_input_size = self.cnn_bottleneck_size
        self.lstm_hidden_size = 1024
        self.lstm_bidirectional = False
        self.lstm_num_dir = 1

        #settings related to training
        self.cnn_lr = 1e-5
        self.rnn_lr = 1e-4
        self.cnn_spec_lr= 1e-4

        self.theta=0.2

        # self.clip = 32
        self.num_epochs=5000
        self.watch_every=100
        self.cnn_dropout=0.2

class Path:
    '''
    path settings
    '''
    def __init__(self,cwd):
        self.anno_path = os.path.join(cwd,'Annotations2')
        self.images_path = os.path.join(cwd,'cropped_aligned1')
        self.specs_path = os.path.join('specs')

        self.train_csv_path = os.path.join(self.anno_path,'Train_Set')
        self.val_csv_path = os.path.join(self.anno_path,'Val_Set')
        self.model_save_path = ''
        self.model_save_path_teacher_arousal='cnnlstmmodel-amb_arousal_58.pt'
        self.model_save_path_teacher_valence=''
        self.model_save_path_v = ''
        self.model_save_path_a = ''
        self.model_save_path_teacher = ''
        self.model_save_path_student_arousal = ''
        self.model_save_path_student_valence = ''
        self.model_save_path_student_mt = ''
     
