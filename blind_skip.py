import torch 
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms

#import torchsample as ts
import torch.backends.cudnn as cudnn
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np



#implement details
'''
z = 3*W*H ~ U(0;1/10)
nu=nd=[8, 16, 32, 64, 128]
ku=kd=[3, 3, 3, 3, 3]
ns=[0, 0, 0, 4, 4]
ks=[NA, NA, NA, 1, 1]
sigma p=1/30
numiter = 2400
LR = 0.01
upsampling=bilinear
'''

class skip_denoise(nn.Module):

    def __init__(self,num_input_channels=32, num_output_channels=3, 
        num_channels_down=[128, 128, 128, 128, 128], num_channels_up=[128, 128, 128, 128, 128], num_channels_skip=[4, 4, 4, 4, 4], 
        filter_size_down=3, filter_size_up=3, filter_skip_size=1,
        need_sigmoid=True, need_bias=True, 
        pad='zero', upsample_mode='bilinear', downsample_mode='stride', act_fun='LeakyReLU', need1x1_up=True):

        super(skip_denoise, self).__init__()

        #class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)

        down = nn.Sequential()
        up = nn.Sequential()
        skip =nn.Sequential()
        skip_exist=[]
        last = len(num_channels_up)-1
        for i in range(len(num_channels_down)):
            to_pad = int((filter_size_up - 1) / 2)
            #down
            if i==0:
                deep1=nn.Sequential(nn.ReflectionPad2d(to_pad),nn.Conv2d(num_input_channels,num_channels_down[i],kernel_size=filter_size_down,stride=2,padding=0,bias=need_bias),nn.BatchNorm2d(num_channels_down[i]),nn.LeakyReLU(0.2, inplace=True))

                deep2=nn.Sequential(nn.ReflectionPad2d(to_pad),nn.Conv2d(num_channels_down[i],num_channels_down[i],kernel_size=filter_size_down,stride=1,padding=0,bias=need_bias),nn.BatchNorm2d(num_channels_down[i]),nn.LeakyReLU(0.2, inplace=True))
            
            elif (i==len(num_channels_up)-1):
                deep1=nn.Sequential(nn.ReflectionPad2d(to_pad),nn.Conv2d(num_channels_down[i-1],num_channels_down[i],kernel_size=filter_size_down,stride=2,padding=0,bias=need_bias),nn.BatchNorm2d(num_channels_down[i]),nn.LeakyReLU(0.2, inplace=True))

                deep2=nn.Sequential(nn.ReflectionPad2d(to_pad),nn.Conv2d(num_channels_down[i],num_channels_down[i],kernel_size=filter_size_down,stride=1,padding=0,bias=need_bias),nn.BatchNorm2d(num_channels_down[i]),nn.LeakyReLU(0.2, inplace=True),nn.Upsample(scale_factor=2, mode=upsample_mode))
            else :

                deep1=nn.Sequential(nn.ReflectionPad2d(to_pad),nn.Conv2d(num_channels_down[i-1],num_channels_down[i],kernel_size=filter_size_down,stride=2,padding=0,bias=need_bias),nn.BatchNorm2d(num_channels_down[i]),nn.LeakyReLU(0.2, inplace=True))

                deep2=nn.Sequential(nn.ReflectionPad2d(to_pad),nn.Conv2d(num_channels_down[i],num_channels_down[i],kernel_size=filter_size_down,stride=1,padding=0,bias=need_bias),nn.BatchNorm2d(num_channels_down[i]),nn.LeakyReLU(0.2, inplace=True))
            #skip
            if (num_channels_skip[i]!=0):
                if i ==0 :
                    skip1 = nn.Sequential(nn.ReflectionPad2d((0,0,0,0)),nn.Conv2d(num_input_channels,num_channels_skip[i],kernel_size=filter_skip_size,stride=1,bias=need_bias),nn.BatchNorm2d(num_channels_skip[i]),nn.LeakyReLU(0.2,inplace=True))
                    self.add_module("s"+str(i+1),skip1)
                    skip_exist.append("True")
                else: 
                    skip1 = nn.Sequential(nn.ReflectionPad2d((0,0,0,0)),nn.Conv2d(num_channels_down[i-1],num_channels_skip[i],kernel_size=filter_skip_size,stride=1,bias=need_bias),nn.BatchNorm2d(num_channels_skip[i]),nn.LeakyReLU(0.2,inplace=True))
                    self.add_module("s"+str(i+1),skip1)
                    skip_exist.append("True")
            
            else:
                skip_exist.append("False")            
        
            
            modules = []
            modules.append(deep1)
            modules.append(deep2)
            sequential = nn.Sequential(*modules)
            self.add_module("d"+str(i+1),sequential)

            if (i==0):#u1
                back1 = nn.Sequential(nn.ReflectionPad2d((0,0,0,0)),nn.Conv2d(num_channels_up[i],num_output_channels,kernel_size=1,stride=1,bias=need_bias),nn.Sigmoid())
            elif (i==1):#u2
                back1 = nn.Sequential(nn.BatchNorm2d(num_channels_up[i]+num_channels_skip[i-1]),nn.ReflectionPad2d(to_pad),nn.Conv2d(num_channels_up[i]+num_channels_skip[i-1],num_channels_up[i-1],kernel_size=3,stride=1,bias=need_bias),nn.BatchNorm2d(num_channels_up[i-1]),nn.LeakyReLU(0.2,inplace=True),nn.ReflectionPad2d((0,0,0,0)),nn.Conv2d(num_channels_up[i-1],num_channels_up[i-1],kernel_size=1,stride=1,bias=need_bias),nn.BatchNorm2d(num_channels_up[i-1]),nn.LeakyReLU(0.2,inplace=True))
            else:#u3 u4 u5
                back1 = nn.Sequential(nn.BatchNorm2d(num_channels_up[i]+num_channels_skip[i-1]),nn.ReflectionPad2d(to_pad),nn.Conv2d(num_channels_up[i]+num_channels_skip[i-1],num_channels_up[i-1],kernel_size=filter_size_up,stride=1,bias=need_bias),nn.BatchNorm2d(num_channels_up[i-1]),nn.LeakyReLU(0.2,inplace=True),nn.ReflectionPad2d((0,0,0,0)),nn.Conv2d(num_channels_up[i-1],num_channels_up[i-1],kernel_size=1,stride=1,bias=need_bias),nn.BatchNorm2d(num_channels_up[i-1]),nn.LeakyReLU(0.2,inplace=True),nn.Upsample(scale_factor=2,mode=upsample_mode) )


            self.add_module("u"+str(i+1),back1)
            
         
        self.middle = nn.Sequential(nn.BatchNorm2d(num_channels_up[last]+num_channels_skip[last]),nn.ReflectionPad2d(to_pad),nn.Conv2d(num_channels_up[last]+num_channels_skip[last],num_channels_up[last],kernel_size=filter_size_up,stride=1,bias=need_bias),nn.BatchNorm2d(num_channels_up[last]),nn.LeakyReLU(0.2,inplace=True),nn.ReflectionPad2d((0,0,0,0)),nn.Conv2d(num_channels_up[last],num_channels_up[last],kernel_size=1,stride=1,bias=need_bias),nn.BatchNorm2d(num_channels_up[last]),nn.LeakyReLU(0.2,inplace=True),nn.Upsample(scale_factor=2,mode=upsample_mode))


        
    def forward(self, x):
        outd1 = self.d1(x)
        outd2 =self.d2(outd1)
        outd3 = self.d3(outd2)
        outd4 = self.d4(outd3)
        outd5 = self.d5(outd4) 
        
        
        
        temp1 = self.s5(outd4)
        temp2 = torch.cat((outd5,temp1),1)
        out = self.middle(temp2)

        out = self.u5(torch.cat((self.s4(outd3),out),1))
        
        out = self.u4(torch.cat((self.s3(outd2),out),1))
            
        out = self.u3(torch.cat((self.s2(outd1),out),1))

        out = self.u2(torch.cat((self.s1(x),out),1))

        out = self.u1(out)

        return out



#net = skip_denoise()

#print (net)

def skip_net(self,num_input_channels=32, num_output_channels=3, 
        num_channels_down=[128, 128, 128, 128, 128], num_channels_up=[128, 128, 128, 128, 128], num_channels_skip=[4, 4, 4, 4, 4], 
        filter_size_down=3, filter_size_up=3, filter_skip_size=1,
        need_sigmoid=True, need_bias=True, 
        pad='zero', upsample_mode='bilinear', downsample_mode='stride', act_fun='LeakyReLU', need1x1_up=True):
    
    model = skip_denoise(num_input_channels, num_output_channels, 
        num_channels_down, num_channels_up, num_channels_skip, 
        filter_size_down=3, filter_size_up=3, filter_skip_size=1,
        need_sigmoid=True, need_bias=True, 
        pad='zero', upsample_mode='bilinear', downsample_mode='stride', act_fun='LeakyReLU', need1x1_up=True)

    return model



