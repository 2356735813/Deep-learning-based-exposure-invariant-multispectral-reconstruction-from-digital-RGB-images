from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
import os

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def initialize_logger(file_dir):
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename=file_dir, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(message)s', "%Y-%m-%d %H:%M:%S")
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)
    return logger

def save_checkpoint(model_path, epoch, iteration, model, optimizer):
    state = {
        'epoch': epoch,
        'iter': iteration,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    torch.save(state, os.path.join(model_path, 'net_%depoch.pth' % epoch))

class Loss_MRAE(nn.Module):
    def __init__(self):
        super(Loss_MRAE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = torch.abs(outputs - label) / label
        mrae = torch.mean(error.view(-1))
        return mrae

class Loss_RMSE(nn.Module):
    def __init__(self):
        super(Loss_RMSE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = outputs-label
        sqrt_error = torch.pow(error,2)
        rmse = torch.sqrt(torch.mean(sqrt_error.view(-1)))
        return rmse

class Loss_PSNR(nn.Module):
    def __init__(self):
        super(Loss_PSNR, self).__init__()

    def forward(self, im_true, im_fake, data_range=255):
        N = im_true.size()[0]
        C = im_true.size()[1]
        H = im_true.size()[2]
        W = im_true.size()[3]
        Itrue = im_true.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
        Ifake = im_fake.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
        mse = nn.MSELoss(reduce=False)
        err = mse(Itrue, Ifake).sum(dim=1, keepdim=True).div_(C * H * W)
        psnr = 10. * torch.log((data_range ** 2) / err) / np.log(10.)
        return torch.mean(psnr)

class Pyr_Loss (nn.Module):
    def __init__(self,weight=1.0):
        super(Pyr_Loss, self).__init__()
        self.weight =weight
        self.criterion = nn.L1Loss(reduction='sum')
    def forward(self,images, labels):
        n = len(images)
        loss = 0
        for m in range(0,n-1):
            loss += self.weight*(2**(n-m-2))*self.criterion(images[m],F.interpolate(labels[m],(images[m].shape[2],images[m].shape[3]),mode='bilinear',align_corners=True))/images[m].shape[0]
        return loss

class Rec_Loss(nn.Module):
    def __init__(self,weight=1):
        super(Rec_Loss, self).__init__()
        self.weight = weight
        self.criterion = nn.L1Loss(reduction='sum')
    def forward(self,images, labels):
        loss = self.weight * self.criterion(images[-1],labels[-1])/images[-1].shape[0]
        return loss

class Adv_loss(nn.Module):
    def __init__(self,size =256 ,weight=1.0):
        super(Adv_loss,self).__init__()
        self.weight = weight
        self.size = size
    def forward(self,P_Y):
        loss = -self.weight * 12 *self.size*self.size*torch.mean(torch.log(torch.sigmoid(P_Y)+1e-9))
        return loss

class My_loss(nn.Module):
    def __init__(self,size =256,Pyr_weight = 1.0,Rec_weight = 1.0, Adv_weight = 1.0):
        super(My_loss,self).__init__()
        self.pyr_loss = Pyr_Loss(Pyr_weight)
        self.rec_loss = Rec_Loss(Rec_weight)
        self.adv_loss = Adv_loss(size,Adv_weight)
    def forward(self,images,labels,P_Y = None,withoutadvloss = False):
        pyrloss =self.pyr_loss(images, labels)
        recloss =self.rec_loss(images,labels)
        if withoutadvloss:
            myloss = pyrloss + recloss
            return recloss,pyrloss,myloss
        else:
            advloss =self.adv_loss(P_Y)
            myloss = pyrloss + recloss + advloss
            return recloss, pyrloss, advloss, myloss

def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename

def record_loss(loss_csv, epoch, iteration, epoch_time, lr, train_loss, test_loss):
    """ Record many results."""
    loss_csv.write('{},{},{},{},{},{}\n'.format(epoch, iteration, epoch_time, lr, train_loss, test_loss))
    loss_csv.flush()
    loss_csv.close