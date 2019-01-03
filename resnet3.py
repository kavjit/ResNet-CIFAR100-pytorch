"""
Created on Tue Oct  2 16:28:13 2018

@author: kavjit
"""
#Note: in this architecture in each block only the first convolution has a stride =2 if it exists. If multiple blocks are run 
#back to back (x4 etc) then only the first block in this series will have the first convolution with stride 2. All other
#convolutions of the 2nd block onwards have stride = 1.
#remember that stride = 2 reduces dimension of image by half
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.model_zoo as model_zoo


import numpy as np
import h5py
from random import randint
import time 

model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth'}

#loading data
transforms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


trainset = torchvision.datasets.CIFAR100(root='./data', train=True,download=True, transform=transforms)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,shuffle=True, num_workers=2)		#numworkers?

testset = torchvision.datasets.CIFAR100(root='./data', train=False,download=True, transform=transforms)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,shuffle=False, num_workers=2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class basic_block(nn.Module):
    def __init__(self,filt,inchannels,channels,stride=1,padding =1,downsample = None):    #recheck passing variables
        super(basic_block, self).__init__()
        self.conv1 = nn.Conv2d(inchannels,channels,kernel_size = filt,padding = padding, stride = stride)
        self.conv2 = nn.Conv2d(channels,channels,kernel_size = filt,padding = padding, stride = 1)
        self.batchnorm = nn.BatchNorm2d(channels)
        self.downsample = downsample
        
    def forward(self,x):
        residual = x
        out = self.conv1(x)
        out = self.batchnorm(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.batchnorm(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return out
        
class resnet(nn.Module):
    def __init__(self,basic_block,layers):  #layers is a list which contains number of layers for each basic block in the whole architecture
        self.inchannels = 32
        super(resnet, self).__init__()
        
        self.conv = nn.Conv2d(3, 32, kernel_size = 3, padding = 1, stride = 1,bias = False)
        self.batchnorm = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(p=0.5)
        
        self.conv2_x = self._make_layer(basic_block,layers[0],1,32)
        self.conv3_x = self._make_layer(basic_block,layers[1],2,64)
        self.conv4_x = self._make_layer(basic_block,layers[2],2,128)
        self.conv5_x = self._make_layer(basic_block,layers[3],2,256)
        
        self.maxpool = nn.MaxPool2d(4, stride = 2)   #ensures that output imzge is 1x1
        self.fc = nn.Linear(256, 100)       ###CORRECTTT??????
        
        
    def _make_layer(self,basic_block,block_count,stride,channels):
        downsample = None
        #print('make_layer')
        if stride!=1 or self.inchannels!=channels:
            downsample = nn.Sequential(nn.Conv2d(self.inchannels,channels,kernel_size = 1,stride = stride, bias=False),nn.BatchNorm2d(channels))

        full_block = []
        block = basic_block(3,self.inchannels,channels,stride,1,downsample)   #this is for the first layer of the entire basic block segment
        full_block.append(block)
        self.inchannels = channels
        for i in range(1,block_count):        
            block = basic_block(3,self.inchannels,channels)              
            full_block.append(block)                   
        
        return nn.Sequential(*full_block)
    
    def forward(self, x):
        x = self.conv(x)    #output of this will have 32 channels
        x = self.batchnorm(x)
        x = F.relu(x)
        x = self.dropout(x)      
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1) #correct?
        x = self.fc(x)   
        return x
         

        
pretrained_var = False #CHANGE THIS TO TRUE TO LOAD PRETRAINED MODEL

def resnet18(pretrained=False, **kwargs):
    model = resnet(basic_block, [2, 4, 4, 2], **kwargs)
    if pretrained:
        model = torchvision.models.resnet.ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2])
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

#Defining the model 
net = resnet18(pretrained = pretrained_var)

#resetting fully connected layer
if pretrained_var:
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 100)

net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr = 0.001)

#defining upsampling for if pretrained = True
upsample = nn.Upsample(scale_factor = 7)

#hyperparameters
epochs = 100


for epoch in range(epochs):  
    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        
        if pretrained_var:  #upsampling to 224 if using pretrained model
            inputs = upsample(inputs)
            
        #print(inputs)
        inputs, labels = inputs.to(device), labels.to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0) #why?
        correct += (predicted == labels).sum().item()
    
    print('accuracy at epoch {} = {}'.format(epoch,correct/total))

    if epoch%10 == 0:
        torch.save(net, 'kavjit_model.ckpt')


print('Finished Training')



#net = torch.load('kavjit_model3.ckpt', map_location = 'cpu')


correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data       
        if pretrained_var:  #upsampling to 224 if using pretrained model
            images = upsample(images)
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)   #output.data? #_for ignoring a value while unpacking second return is argmax index, 1 to indicate axis to check
        total += labels.size(0) #why?
        correct += (predicted == labels).sum().item()


print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

