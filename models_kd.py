
from cmath import tanh
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
import os
from torchvision import models
from config1 import ConfigBasic
import torch.nn as nn
import torch
from config import ConfigCNNLSTM


cfg=ConfigCNNLSTM()
# device='cpu'
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
rnn_type = 'gru'
rnn_layer = 1
rnn_input_size=512
rnn_hidden_size = cfg.lstm_hidden_size
bidirectional = False
output_size = 8
rnn_num_dir=1

learning_rate_spec_cnn=cfg.cnn_spec_lr
learning_rate_cnn=cfg.cnn_lr
learning_rate_rnn=cfg.rnn_lr


cfgvgg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


#### Audio backbone###

class CNN_Spec_Backbone(nn.Module):
    def __init__(self,cfg):
        super(CNN_Spec_Backbone,self).__init__()      
        self.cnn_spec_backbone = models.resnet18(pretrained=True)
        self.cnn_spec_backbone = nn.Sequential(*list(self.cnn_spec_backbone.children())[:-1])

    def forward(self,x):
        bsz = len(x)
        out = self.cnn_spec_backbone(x)
        # print(out.shape)
        out = out.view(bsz,-1) # used of res18 or res18_pre
        #print(out.shape)
        # out = self.drop(out)
        return out


#### Audio Only Model  #####

class Spec_fc(nn.Module):
    def __init__(self):
        super(Spec_fc,self).__init__()
        self.out_layer1 = nn.Linear(512,256)
        self.out_layer2 = nn.Linear(256,128)
        self.out_layer3 = nn.Linear(128,64)
        self.out_layer4 = nn.Linear(64,1)
        
    def forward(self,x):
        output = self.out_layer1(x)
        output = self.out_layer2(output)
        intermediate_output_student = torch.relu(output)
        output = self.out_layer3(output)
        output = self.out_layer4(output)
        
        return intermediate_output_student ,output







class CNN_Spec_fc(nn.Module):

    def __init__(self,cfg):
        super(CNN_Spec_fc,self).__init__()
        self.cnn_spec=CNN_Spec_Backbone(cfg)
        self.cnn_fc=Spec_fc()

        self.cnn_spec_optimizer = optim.Adam(self.cnn_spec.parameters(),lr=learning_rate_spec_cnn)
        self.cnn_fc_optimizer = optim.Adam(self.cnn_fc.parameters(),lr=learning_rate_cnn)


        self.cnn_spec = self.cnn_spec.to(device)
        self.cnn_fc=self.cnn_fc.to(device)
        self.cfg = cfg
        

    def model_out(self,specs):
        output=[]
        int_out_student=[]
        self.cnn_spec_optimizer.zero_grad()
        h0 = torch.zeros(rnn_layer*rnn_num_dir,1,rnn_hidden_size).to(device)
        
        c0 = torch.zeros(rnn_layer*rnn_num_dir,1,rnn_hidden_size).to(device)
        for spec in specs:
            
            cnn_out=self.cnn_spec(spec)
            int_out, final_out = self.cnn_fc(cnn_out)

            # output.append(final_out.squeeze(1))
            output.append(final_out)
            int_out_student.append(int_out)



        output= torch.stack(output) 
        int_out_student=torch.stack(int_out_student)
        return int_out_student, output

    def predict(self,specs):
        cnn_spec_out = self.cnn_spec(specs)
        int_out,final_out = self.cnn_fc(cnn_spec_out)
        return final_out





#### Visual Backbone #####

class CNN_Backbone(nn.Module):
    

    
    def __init__(self,cfg):
        super(CNN_Backbone,self).__init__()      
        # self.cnn_backbone = models.resnet18(pretrained=True)

        # model_path = os.path.join('Pre_trained_models','affectNetRes18_small.pt')
        # cfg1=ConfigBasic()
        # self.cnn_backbone=BasicModel(cfg1)
        # checkpoint = torch.load(model_path,map_location=cfg1.device)
        # self.cnn_backbone.load_state_dict(checkpoint,strict=True)
        # self.cnn_backbone = nn.Sequential(*list(self.cnn_backbone.children())[:-2])


        # checkpoint = torch.load(model_path,map_location=cfg.device)
        # self.cnn_backbone.load_state_dict(checkpoint,strict=True)
        # self.cnn_backbone = nn.Sequential(*list(self.cnn_backbone.children())[:-1])
        


        model_path = os.path.join('Pre_trained_models','PrivateTest_model.t7')

        self.cnn_backbone =VGG('VGG19')
        checkpoint=torch.load(model_path,map_location=device)
        del checkpoint['net']['classifier.weight']
        del checkpoint['net']['classifier.bias']
        self.cnn_backbone.load_state_dict(checkpoint['net'],strict=False)

        self.cnn_backbone.to(device)
        

            
        # if cfg.cnn_backbone_type == 'mobilefacenet':
        #     self.cnn_backbone = MobileFaceNet([112, 112],cfg.cnn_bottleneck_size)   
        #     checkpoint = torch.load('Pre_trained_models/mobilefacenet_model_best.pth.tar')      
        #     #print('Use MobileFaceNet as backbone') 
        #     self.cnn_backbone.load_state_dict(checkpoint['state_dict'])
            
        # if cfg.cnn_backbone_type == 'vit':
        #     raise NotImplementedError("Not implemented yet.")

        # if cfg.cnn_backbone_type == 'dense121':
        #     raise NotImplementedError("Not implemented yet.")

        # if cfg.cnn_backbone_type == 'dense169':
        #     raise NotImplementedError("Not implemented yet.")

            
        self.drop = nn.Dropout(cfg.cnn_dropout)
        
    def forward(self,x):
        bsz = len(x)
        out = self.cnn_backbone(x)
        # print(out.shape)
        out = out.view(bsz,-1) # used of res18 or res18_pre
        #print(out.shape)
        out = self.drop(out)
        return out













class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfgvgg[vgg_name])
        self.fc = nn.Linear(4608, 512)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = F.dropout(out, p=0.2, training=self.training)
        
        out = self.fc(out)
        return out

    def _make_layers(self, cfgvgg):
        layers = []
        in_channels = 3
        for x in cfgvgg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class TemporalLSTM_t(nn.Module):
    '''
    LSTM part
    inputs: input, (h0,c0)
            input: (seq_len,batch,input_size)
            h_0: (num_layers*num_directions,batch,hidden_size)
            c_0: (num_layers*num_directions,batch,hidden_size)
    outputs: output, (h_n,c_n)
            output: (seq_len,batch,num_directions*hidden_size)
            h_n: (num_layers*num_directions,batch,hidden_size)
            c_n: (num_layers*num_directions,batch,hidden_size)
    '''
    def __init__(self,cfg):
        super(TemporalLSTM_t,self).__init__()
        '''
        cfg: Configuration Class. Follow config.py
        
        '''
        self.rnn_type = 'gru'
        self.layer = 1
        self.input_size = 512
        self.hidden_size = cfg.lstm_hidden_size
        self.bidirectional = False
        self.output_size =1
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.input_size,self.hidden_size,num_layers=self.layer,bidirectional=self.bidirectional)
        if self.rnn_type == 'gru':
            self.rnn = nn.GRU(self.input_size,self.hidden_size,num_layers=self.layer,bidirectional=self.bidirectional)
            
        self.out_layer1 = nn.Linear(self.hidden_size,1024)
        self.out_layer2 = nn.Linear(self.hidden_size+512,256)
        self.out_layer3 = nn.Linear(256,128)
        self.out_layer4 = nn.Linear(128,self.output_size)
        
    def forward(self,x,h0,c0,spec):
        if self.rnn_type == 'gru':
            
            rnn_out, h0 = self.rnn(x,h0)
            c0 = h0
        
        if self.rnn_type == 'lstm':
            rnn_out, (h0,c0) = self.rnn(x,(h0,c0))
        #concatenate
        output = self.out_layer1(rnn_out)
        output = torch.cat((output,spec),2)
        output = self.out_layer2(output)
        # output = torch.relu(output)
        intermediate_output_teacher=self.out_layer3(output)

        # output = self.out_layer3(output)
        # intermediate_output_teacher = torch.relu(output)
        output = self.out_layer4(intermediate_output_teacher)
    
        return intermediate_output_teacher,output, rnn_out, (h0,c0)




class CNN_LSTM_Teacher(nn.Module):
    '''
    CNN-LSTM model.
    
    '''
    def __init__(self,cfg):
        super(CNN_LSTM_Teacher,self).__init__()
        self.cnn=CNN_Backbone(cfg)

        self.rnn = TemporalLSTM_t(cfg)
        self.spec_cnn=CNN_Spec_Backbone(cfg)
        self.spec_cnn_optimizer = optim.Adam(self.spec_cnn.parameters(),lr=learning_rate_spec_cnn)
        self.cnn_optimizer = optim.Adam(self.cnn.parameters(),lr=learning_rate_cnn)
        self.rnn_optimizer = optim.Adam(self.rnn.parameters(),lr=learning_rate_rnn)

        self.cnn = self.cnn.to(device)
        self.rnn = self.rnn.to(device)
        self.spec_cnn= self.spec_cnn.to(device)
        self.cfg = cfg
        


    def model_out(self,features,specs):
        output=[]
        output_spec=[]
        intermediate_output_teacher=[]
        self.cnn_optimizer.zero_grad()
        self.rnn_optimizer.zero_grad()
        self.spec_cnn_optimizer.zero_grad()

        h0 = torch.zeros(rnn_layer*rnn_num_dir,1,rnn_hidden_size).to(device)
        c0 = torch.zeros(rnn_layer*rnn_num_dir,1,rnn_hidden_size).to(device)
        i=0
        for feature in features:
            cnn_out=self.cnn(feature).unsqueeze(1)
            spec_out=self.spec_cnn(specs[i]).unsqueeze(1)
            i+=1
            int_out_teacher,final_out, _ , _ = self.rnn(cnn_out,h0,c0,spec_out)
            output.append(final_out.squeeze(1))
            intermediate_output_teacher.append(int_out_teacher.squeeze(1))

        output= torch.stack(output)
        intermediate_output_teacher=torch.stack(intermediate_output_teacher) 
        return intermediate_output_teacher,output

    def model_out_val(self,features):
        output=[]
        self.cnn_optimizer.zero_grad()
        self.rnn_optimizer.zero_grad()
        h0 = torch.zeros(rnn_layer*rnn_num_dir,1,rnn_hidden_size).to(device)
        c0 = torch.zeros(rnn_layer*rnn_num_dir,1,rnn_hidden_size).to(device)
        for feature in features.unsqueeze(1):
            cnn_out=self.cnn(feature).unsqueeze(1)
            final_out, _ , _ = self.rnn(cnn_out,h0,c0)
            output.append(final_out.squeeze(1))

        output= torch.stack(output) 
        return output    

    def predict(self,features,h0,c0,sp):
        cnn_out = self.cnn(features).unsqueeze(1)
        spec_out=self.spec_cnn(sp).unsqueeze(1)
        if self.cfg.rnn_type == 'gru':
            _,final_out, _,(h0,c0) = self.rnn(cnn_out,h0,h0,spec_out)
        if self.cfg.rnn_type == 'lstm':
            final_out, _ , (h0,c0) = self.rnn(cnn_out,h0[-1].unsqueeze(0),c0[-1].unsqueeze(0))
        return final_out,h0,c0
        

class TemporalLSTM_s(nn.Module):
    '''
    LSTM part
    inputs: input, (h0,c0)
            input: (seq_len,batch,input_size)
            h_0: (num_layers*num_directions,batch,hidden_size)
            c_0: (num_layers*num_directions,batch,hidden_size)
    outputs: output, (h_n,c_n)
            output: (seq_len,batch,num_directions*hidden_size)
            h_n: (num_layers*num_directions,batch,hidden_size)
            c_n: (num_layers*num_directions,batch,hidden_size)
            
    GRU part
            
    
    '''
    def __init__(self):
        super(TemporalLSTM_s,self).__init__()
        '''
        cfg: Configuration Class. Follow config.py
        
        '''
        self.rnn_type = 'gru'
        self.layer = 1
        self.input_size = 512
        self.hidden_size = cfg.lstm_hidden_size
        self.bidirectional = False
        self.output_size =1
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.input_size,self.hidden_size,num_layers=self.layer,bidirectional=self.bidirectional)
        if self.rnn_type == 'gru':
            self.rnn = nn.GRU(self.input_size,self.hidden_size,num_layers=self.layer,bidirectional=self.bidirectional)
            
        self.out_layer1 = nn.Linear(self.hidden_size,512)
        self.out_layer2 = nn.Linear(512,256)
        self.out_layer3 = nn.Linear(256,128)
        self.out_layer4 = nn.Linear(128,64)
        self.out_layer5 = nn.Linear(64,self.output_size)
        
    def forward(self,x,h0,c0):
        if self.rnn_type == 'gru':
            
            rnn_out, h0 = self.rnn(x,h0)
            c0 = h0
        
        if self.rnn_type == 'lstm':
            rnn_out, (h0,c0) = self.rnn(x,(h0,c0))
        
        output = self.out_layer1(rnn_out)
        output = self.out_layer2(output)

        intermediate_output_student= self.out_layer3(output)

        
        output = self.out_layer4(intermediate_output_student)
        output = self.out_layer5(output)
        
        return intermediate_output_student,output, rnn_out, (h0,c0)

class CNN_LSTM_s(nn.Module):
    '''
    CNN-LSTM model.
    
    '''
    def __init__(self,cfg):
        super(CNN_LSTM_s,self).__init__()
        # self.cnn=VGG('VGG19').to(device)
        self.cnn=CNN_Backbone(cfg)

        self.rnn = TemporalLSTM_s()
        self.cnn_optimizer = optim.Adam(self.cnn.parameters(),lr=learning_rate_cnn)
        self.rnn_optimizer = optim.Adam(self.rnn.parameters(),lr=learning_rate_rnn)

        self.cnn = self.cnn.to(device)
        self.rnn = self.rnn.to(device)
        self.cfg = cfg
        


    def model_out(self,features):
        output=[]
        int_out_student=[]
        self.cnn_optimizer.zero_grad()
        self.rnn_optimizer.zero_grad()
        h0 = torch.zeros(rnn_layer*rnn_num_dir,1,rnn_hidden_size).to(device)
        
        c0 = torch.zeros(rnn_layer*rnn_num_dir,1,rnn_hidden_size).to(device)
        for feature in features:
            
            cnn_out=self.cnn(feature).unsqueeze(1)
            intermediate_out_student,final_out, _ , _ = self.rnn(cnn_out,h0,c0)

            output.append(final_out.squeeze(1))
            int_out_student.append(intermediate_out_student.squeeze(1))

        output= torch.stack(output) 
        int_out_student=torch.stack(int_out_student)
        return int_out_student, output

    def model_out_val(self,features):
        output=[]
        self.cnn_optimizer.zero_grad()
        self.rnn_optimizer.zero_grad()
        h0 = torch.zeros(rnn_layer*rnn_num_dir,1,rnn_hidden_size).to(device)
        c0 = torch.zeros(rnn_layer*rnn_num_dir,1,rnn_hidden_size).to(device)
        for feature in features.unsqueeze(1):
            cnn_out=self.cnn(feature).unsqueeze(1)
            final_out, _ , _ = self.rnn(cnn_out,h0,c0)
            output.append(final_out.squeeze(1))

        output= torch.stack(output) 
        return output    

    def predict(self,features,h0,c0):
        cnn_out = self.cnn(features).unsqueeze(1)
        if self.cfg.rnn_type == 'gru':
            inter_out,final_out, _,(h0,c0) = self.rnn(cnn_out,h0,h0)
        if self.cfg.rnn_type == 'lstm':
            final_out, _ , (h0,c0) = self.rnn(cnn_out,h0[-1].unsqueeze(0),c0[-1].unsqueeze(0))
        return final_out,h0,c0
        

