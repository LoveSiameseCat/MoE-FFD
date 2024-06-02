import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import random
import os
import torch
import math
import albumentations as alb
from albumentations.pytorch.transforms import ToTensorV2
import copy

img_size=224
mean = [0.5,0.5,0.5]
std = [0.5,0.5,0.5]


base_transform = alb.Compose([
    alb.Resize(img_size,img_size),
    alb.Normalize(mean=mean, std=std),
    ToTensorV2(),
])

class FFPP_Dataset(Dataset):
    def __init__(self,path,frame=20,phase='train'):
        super(FFPP_Dataset, self).__init__()
        assert phase in ['train','valid','test']
        self.path = path
        self.frame = frame
        self.phase = phase
        self.list = self.ff_generate_list()
        self.images = [line.strip().split()[0] for line in self.list]
        self.labels = [line.strip().split()[1] for line in self.list]

    def ff_generate_list(self):
        list_ = []
        method = os.listdir(self.path)
        for m in method:
            if m == 'original':
                if self.phase == 'train':
                    frame = 4*self.frame # To do balance
                else:
                    frame = self.frame
                label = 0
            else:
                frame = self.frame
                label = 1
            method_path = os.path.join(self.path,m)
            for v in os.listdir(method_path):
                v_path = os.path.join(method_path,v)
                pic = os.listdir(v_path)
                pic.sort(key=lambda x:int(x[0:-4]))
                if len(pic) < frame:
                    pic = pic * (frame//len(pic)+1)
                interval = len(pic)//frame
                imgs_path = [os.path.join(v_path,pic[i*interval])+' '+str(label) + '\n' for i in range(frame)]
                list_ += imgs_path
        return list_


    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        fn, label = self.images[item],self.labels[item]
        img =  cv2.cvtColor(cv2.imread(fn),cv2.COLOR_BGR2RGB)
        img = base_transform(image=img)['image']
        return img,int(label)

'''
eval dataset
'''

class TestDataset(Dataset):
    def __init__(self, path,dataset,frame=20): # ['CS',0]
        super(TestDataset, self).__init__()
        assert dataset in ['FFPP','CDF','DFDC_P']
        # self.path = path
        self.frame = frame
        if dataset == 'FFPP':
            self.list = self.ff_video_gen(path)
        elif dataset == 'DFDC_P':
            self.list = self.DFDCP_video_gen(path)
        elif dataset == 'CDF':
            self.list = self.celeb_video_gen(path)
        else:
            raise NotImplemented('False Dataset!')

    def __len__(self):
        return len(self.list)

    def ff_video_gen(self,path):
        list_ = []
        method = os.listdir(path)
        for m in method:
            label = 0 if m == 'original' else 1
            method_path = os.path.join(path,m)
            for v in os.listdir(method_path):
                v_path = os.path.join(method_path,v)
                list_.append(v_path+' '+str(label)+'\n')
        return list_

    def celeb_video_gen(self,path):
        list_ = []
        method = os.listdir(path)
        for m in method:
            if 'real' in m:
                label = 0
            else:
                label = 1
            method_path = os.path.join(path,m)
            for v in os.listdir(method_path):
                v_path = os.path.join(method_path,v)
                list_.append(v_path+' '+str(label)+'\n')
        return list_


    def DFDCP_video_gen(self,path):
        list_ = []
        method = os.listdir(path)
        for m in method:
            if 'real' in m:
                label = 0
            else:
                label = 1
            method_path = os.path.join(path,m)
            for v in os.listdir(method_path):
                v_path = os.path.join(method_path,v)
                if len(os.listdir(v_path))<20:
                    continue
                list_.append(v_path+' '+str(label)+'\n')
        return list_

    def one_image_gen(self,path):
        img =  cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)
        img = base_transform(image=img)['image']
        return img.unsqueeze(0)

    def video_imgs_gen(self,path):
        img_list = []
        pic = os.listdir(path)
        pic.sort(key=lambda x:int(x[0:-4]))
        if len(pic) < self.frame:
            pic = pic * (self.frame//len(pic)+1)
        interval = len(pic)//self.frame
        for i in range(self.frame):
            img_path = os.path.join(path,pic[i*interval])
            img = self.one_image_gen(img_path)
            img_list.append(img)
        return torch.cat(img_list,dim=0)

    def __getitem__(self, item):
        v_path,v_label = self.list[item].strip().split()
        video_img = self.video_imgs_gen(v_path)
        return video_img, int(v_label)


