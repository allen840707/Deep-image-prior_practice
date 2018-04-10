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

from blind_skip import*
from util import*
from downsampler import*
from tensorboardX import SummaryWriter
from skimage.measure import compare_psnr

writer = SummaryWriter("task3/super-resolution22")


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

fname = 'images/LowResolution.png'
trueimage = 'images/SR_GT.png'
INPUT = "noise"
#pad = "reflection"
reg_noise_std = 0.03 #1./30. 
LR = 0.01
#OPTIMIZER="adam"
input_depth = 32
num_iteration = 2000
imsize = -1 # no need to crop
show_every= 22222
figsize=4
PLOT=True
sigma = 25
sigma_ = sigma/255.


factor = 4 

downsampler = Downsampler(n_planes=3, factor=4, kernel_type='lanczos2', phase=0.5, preserve_size=True)

net = skip_denoise(
                input_depth, 3, 
                num_channels_down = [128, 128, 128, 128, 128], 
                num_channels_up   = [128, 128, 128, 128, 128],
                num_channels_skip = [4, 4, 4, 4, 4], 
                upsample_mode='bilinear',
                need_sigmoid=True, need_bias=True)


#true_image = Image.open(trueimage)
true_image_pil = crop_image(get_image(trueimage,imsize)[0],d=32)
true_image_np = pil_to_np(true_image_pil)
print(net)
#crop image


img_noisy_pil = Image.open(fname)
#img_noisy_pil = crop_image(get_image(fname, imsize)[0], d=32)
img_noisy_np = pil_to_np(img_noisy_pil)
#print(img_noisy_np.shape)




#img_pil = img_noisy_pil
#img_np = img_noisy_np


if PLOT==True:
    plot_image_grid([true_image_np], 1, 4)
    plot_image_grid([img_noisy_np],1,4)
    #print ("image size",img_noisy_pil.size)
    #true_image_pil.show()
    #img_noisy_pil.show()

#net input add noise 
#img_noisy_np = np.clip(img_np + np.random.normal(scale=sigma_, size=img_np.shape), 0, 1).astype(np.float32)

net_input = get_noise(input_depth, INPUT, (true_image_pil.size[1], true_image_pil.size[0])).detach()



#convert to variable
img_noisy_var = np_to_var(img_noisy_np)
print(img_noisy_np.shape)

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


#if PLOT==True:
#    plot_image_grid([true_image_np,img_noisy_np], 4, 5)


for j in range(num_iteration):


    if reg_noise_std > 0:
        net_input.data = net_input_saved + (noise.normal_() * reg_noise_std)


    if (torch.cuda.is_available()==True):
        inputs = net_input.cuda()
        fz = img_noisy_var.cuda()
        downsampler = downsampler.cuda() 

    out_HR = net(inputs)
    #print (out_HR)
    out_LR = downsampler(out_HR)
    #print(out_LR)
    optimizer.zero_grad()

    total_loss = criterion(out_LR,fz)
    
    print ('Iteration %05d    Loss %f' % (j, total_loss.data[0]), '\r', end='')
    if  (PLOT==True and j % show_every == 0):
        out_np = var_to_np(out_HR)
        plot_image_grid([np.clip(out_np, 0, 1)], factor=figsize, nrow=1)
        


    
    if (j==1999):
        out_np = var_to_np(out_HR)
        plot_image_grid([np.clip(out_np, 0, 1)], factor=figsize, nrow=1)
        out_pil = np_to_pil(out_np)
        out_pil.save("s_resolution_result.png")
        
        PSNR = compare_psnr(true_image_np,out_np)
        print("\nPSNR =",PSNR)
     


    #plt.figure()
    #plt.plot(j+1,total_loss.data[0])
    writer.add_scalar("mse_loss",total_loss.data[0],j)
    
    #PSNR = compare_psnr(true_image,out_np)    
    #print("PSNR =",PSNR)
    #writer.add_scalar("PSNR",PSNR,j)


 
    total_loss.backward()    
    optimizer.step()

#plt.savefig("./Result/image_add_noise.png")
    

