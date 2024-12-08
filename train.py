# -*- coding:utf-8 -*-
# -----------------------------------------
#   Filename: model.py
#   Author  : Youshen Xiao
#   Email   : xiaoysh2023@shanghaitech.edu.cn
#   Date    : 2024/12/6
# -----------------------------------------
import torch
import model
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
        # writer = SummaryWriter('./log')

        # -----------------------
        # parameters settings
        # -----------------------
        parser = argparse.ArgumentParser()

        parser.add_argument('-input_path', type=str, default='img_2.npy', help='the original image path.')
        parser.add_argument('-mask_path',type=str,default='mask_img_2.npy',help=' mask path.')

        # about training hyper-parameters
        parser.add_argument('-lr', type=float, default=1e-4, help='the initial learning rate.')
        parser.add_argument('-epoch', type=int, default=50001, dest='epoch', help='the total number of epochs for training')
        parser.add_argument('-summary_epoch', type=int, default=200, dest='summary_epoch', help='the current model will be saved per summary_epoch')
        parser.add_argument('-gpu', type=int, default=1, dest='gpu', help='the number of GPU.')
        args = parser.parse_args()


        # -----------------------
        # load data
        # -----------------------
        img_ori=np.load(args.input_path)
        mask=np.load(args.mask_path)
        # img_ori=np.zeros((128,128))
        

        # -----------------------
        # define input coordinates
        # -----------------------
        x=np.linspace(0,1,img_ori.shape[0])
        y=np.linspace(0,1,img_ori.shape[1])
        xs,ys=np.meshgrid(x,y,indexing='ij')
        coor=np.stack([xs.flatten(),ys.flatten()],axis=1)
        B = np.random.randn(2, 512) * 10
        
        
        # -----------------------
        # model & optimizer
        # -----------------------
        DEVICE = torch.device('cuda:{}'.format(str(args.gpu) if torch.cuda.is_available() else 'cpu'))
        mlp = model.MLP(depth=3,mapping_size=1024,hidden_size=256).to(DEVICE)
        loss_fun = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(params=mlp.parameters(), lr=args.lr)

        # put data to gpu/cpu
        img_ori = torch.tensor(img_ori).to(DEVICE).float()
        mask = torch.tensor(mask).to(DEVICE).float()
        coor = torch.tensor(coor).to(DEVICE).float()
        B = torch.tensor(B).to(DEVICE).float()
        pos=model.map_x(coor,B)

        # -----------------------
        # training & validation
        # -----------------------
        mlp.train()
        for i in range(args.epoch):
                img_pre=mlp(pos)
                img_pre = torch.reshape(img_pre, (img_ori.size(0), img_ori.size(1)))
                loss=loss_fun(torch.mul(img_pre,mask),torch.mul(img_ori,mask))
                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # record and print loss
                current_lr = optimizer.state_dict()['param_groups'][0]['lr']
                print('(TRAIN) Epoch[{}/{}]], Lr:{}, Loss:{:.10f}'.format(i + 1,args.epoch, current_lr, loss.item()))
                if i % args.summary_epoch == 0 :
                        np.save(f'result/{i+1}.npy',img_pre.cpu().detach().numpy())
                        image = Image.fromarray((255*img_pre.cpu().detach().numpy()).astype(np.uint8))
                        image.save(f'png/{i+1}.png')






