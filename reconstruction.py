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

from skip import*
from util import*

from tensorboardX import SummaryWriter

writer = SummaryWriter("task1/shuffle")


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
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True

fname = 'images/req1.jpg'
INPUT = "noise"
#pad = "reflection"
reg_noise_std = 1./30. 
LR = 0.01
#OPTIMIZER="adam"
input_depth = 3
num_iteration = 2400
imsize = -1 # no need to crop
show_every=2399
figsize=5
PLOT=True

net = skip_denoise(
                input_depth, 3, 
                num_channels_down = [8, 16, 32, 64, 128], 
                num_channels_up   = [8, 16, 32, 64, 128],
                num_channels_skip = [0, 0, 0, 4, 4], 
                upsample_mode='bilinear',
                need_sigmoid=True, need_bias=True)

print(net)
#crop image 
img_noisy_pil = crop_image(get_image(fname, imsize)[0], d=32)
img_noisy_np = pil_to_np(img_noisy_pil)

img_pil = img_noisy_pil
img_np = img_noisy_np


if PLOT==True:
    plot_image_grid([img_np], 4, 5)
    #print ("image size",img_noisy_pil.size)
    #img_noisy_pil.show()

#net input add noise 
#img_noisy_np = np.clip(img_np + np.random.normal(scale=0.1, size=img_np.shape), 0, 1).astype(np.float32)



for i in range(3):
    for j in range(img_np.shape[1]):    
        np.random.shuffle(img_noisy_np[i][j])


#np.random.shuffle(img_np)

#img_noisy_np = np.random.uniform(0,1,size=img_noisy_np.shape).astype(np.float32)

#img_pil = np_to_pil(img_np)

#img_noisy_pil = np_to_pil(img_noisy_np)  #image+noise



#01 noise input 
net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).detach()


# Compute number of parameters
#s  = sum([np.prod(list(p.size())) for p in net.parameters()]); 
#print ('Number of params: %d' % s)

#convert to variable
img_noisy_var = np_to_var(img_noisy_np)

#Loss&optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=LR)
        

#save 
net_input_saved = net_input.data.clone()
noise = net_input.data.clone()

#cuda

if torch.cuda.is_available():
    print("net is available")
    net.cuda()
    criterion.cuda()
    cudnn.benchmark=True


#plt.figure()
#plt.ylabel('MSE_loss')
#plt.xlabel("iteration")


if PLOT==True:
    plot_image_grid([img_noisy_np], 4, 5)


for j in range(num_iteration):


    if reg_noise_std > 0:
        net_input.data = net_input_saved + (noise.normal_() * reg_noise_std)


    if (torch.cuda.is_available()==True):
        inputs = net_input.cuda()
        fz = img_noisy_var.cuda()
    

    output = net(inputs)
    optimizer.zero_grad()
    total_loss = criterion(output,fz)
    
    print ('Iteration %05d    Loss %f' % (j, total_loss.data[0]), '\r', end='')
    if  (PLOT==True and j % show_every == 0):
        out_np = var_to_np(output)
        plot_image_grid([np.clip(out_np, 0, 1)], factor=figsize, nrow=1)


    #plt.figure()
    #plt.plot(j+1,total_loss.data[0])
   
    writer.add_scalar("mse_loss",total_loss.data[0],j)

    
    total_loss.backward()    
    optimizer.step()

#plt.savefig("./Result/image_add_noise.png")
    

