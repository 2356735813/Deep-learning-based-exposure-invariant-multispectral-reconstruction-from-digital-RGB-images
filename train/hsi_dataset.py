from torch.utils.data import Dataset
import numpy as np
import random
import cv2
import h5py
import glob
import torch
import os
from decomposition import lplas_decomposition as decomposition 

def populate_train_list(bgr_data_path, hyper_data_path):
    image_list_lowlight = glob.glob(bgr_data_path + '/*')
    hyper_list_normal = glob.glob(hyper_data_path + '/*')
    hyper_list_normal = [item for item in hyper_list_normal for _ in range(5)]
    image_list_lowlight.sort()
    hyper_list_normal.sort()
    
    print("image_list_lowlight length:", len(image_list_lowlight))
    print("hyper_list_normal length:", len(hyper_list_normal))
    
    if len(image_list_lowlight) != len(hyper_list_normal):
        print('Data length Error')
        exit()
    
    return image_list_lowlight, hyper_list_normal
    
class TrainDataset(Dataset):
    def __init__(self, data_root, crop_size, arg=True, bgr2rgb=True, stride=8):
        self.crop_size = crop_size
        self.hyper_paths = []
        self.bgr_paths = []
        self.arg = arg
        h,w = 482,512  # img shape
        self.stride = stride
        self.patch_per_line = (w-crop_size)//stride+1
        self.patch_per_colum = (h-crop_size)//stride+1
        self.patch_per_img = self.patch_per_line*self.patch_per_colum

        hyper_data_path = f'{data_root}/Train_Spec/'
        # bgr_data_path = f'{data_root}/standard_Train/'
        bgr_data_path = f'{data_root}/random_Train/'
        
        self.bgr_list,self.hyper_list = populate_train_list(bgr_data_path,hyper_data_path)
        print(f'len(hyper) of ntire2022 dataset:{len(self.hyper_list)}')
        print(f'len(bgr) of ntire2022 dataset:{len(self.bgr_list)}')
        for i in range(len(self.hyper_list)):
            hyper_path = self.hyper_list[i]
            bgr_path = self.bgr_list[i]
            bgrbasename = os.path.basename(bgr_path)
            hypertbasename = os.path.basename(hyper_path)
            new_file_name = '_'.join(bgrbasename.split('_')[:-1])
            print(hypertbasename.split('.')[0])
            print(new_file_name)
            assert hypertbasename.split('.')[0] == new_file_name, 'Hyper and RGB come from different scenes.'
            
            self.hyper_paths.append(hyper_path)
            self.bgr_paths.append(bgr_path)
            print(f'Ntire2022 scene {i} is loaded.')
        self.img_num = len(self.hyper_paths)
        self.length = self.patch_per_img * self.img_num

    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, idx):
        stride = self.stride
        crop_size = self.crop_size
        img_idx, patch_idx = idx//self.patch_per_img, idx%self.patch_per_img
        h_idx, w_idx = patch_idx//self.patch_per_line, patch_idx%self.patch_per_line
        bgr_path = self.bgr_paths[img_idx]
        hyper_path = self.hyper_paths[img_idx]
        with h5py.File(hyper_path, 'r') as mat:
            hyper =np.float32(np.array(mat['cube']))
            hyper = np.transpose(hyper, [0, 2, 1])
        hyper = hyper[:, h_idx * stride:h_idx * stride + crop_size,w_idx * stride:w_idx * stride + crop_size]
        bgr = cv2.imread(bgr_path)
        bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        bgr = np.float32(bgr)
        bgr = (bgr - bgr.min()) / (bgr.max() - bgr.min()+0.0001)
        bgr = np.transpose(bgr, [2, 0, 1])
        bgr = bgr[:,h_idx*stride:h_idx*stride+crop_size, w_idx*stride:w_idx*stride+crop_size]
        rotTimes = random.randint(0,3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        if self.arg:
            bgr = self.arguement(bgr, rotTimes, vFlip, hFlip)
            hyper = self.arguement(hyper, rotTimes, vFlip, hFlip)
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self):
        return self.patch_per_img*self.img_num
        

class ValidDataset(Dataset):
    def __init__(self, data_root, bgr2rgb=True):
        self.hyper_paths = []
        self.bgr_paths = []
        hyper_data_path = f'{data_root}/Val_Spec/'
        # bgr_data_path = f'{data_root}/standard_Val/'
        bgr_data_path = f'{data_root}/random_Val/'
        self.bgr_list,  self.hyper_list = populate_train_list(bgr_data_path,hyper_data_path)
        print(f'len(hyper) of ntire2022 dataset:{len(self.hyper_list)}')
        print(f'len(bgr) of ntire2022 dataset:{len(self.bgr_list)}')
        # for i in range(0,1):
        for i in range(len(self.hyper_list)):
            hyper_path = self.hyper_list[i]
            bgr_path = self.bgr_list[i]
            bgrbasename = os.path.basename(bgr_path)
            hypertbasename = os.path.basename(hyper_path)
            new_file_name = '_'.join(bgrbasename.split('_')[:-1])
            # assert hypertbasename.split('.')[0] == new_file_name, 'Hyper and RGB come from different scenes.'
            self.hyper_paths.append(hyper_path)
            self.bgr_paths.append(bgr_path)
            print(f'Ntire2022 scene {i} is loaded.')

    def __getitem__(self, idx):
        hyper_path = self.hyper_paths[idx]
        bgr_path = self.bgr_paths[idx]
        bgr = cv2.imread(bgr_path)
        bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        bgr = np.float32(bgr)
        bgr = (bgr - bgr.min()) / (bgr.max() - bgr.min()+0.0001)
        bgr = np.transpose(bgr, [2, 0, 1])
        with h5py.File(hyper_path, 'r') as mat:
            hyper =np.float32(np.array(mat['cube']))
            hyper = np.transpose(hyper, [0, 2, 1])
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self):
        return len(self.hyper_paths)