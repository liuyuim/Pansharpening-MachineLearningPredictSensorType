# ------------------------------------------------------------------
# Copyright (c) 2021, Zi-Rong Jin, Tian-Jing Zhang, Cheng Jin, and 
# Liang-Jian Deng, All rights reserved.
#
# This work is licensed under GNU Affero General Public License
# v3.0 International To view a copy of this license, see the
# LICENSE file.
#
# This file is running on WorldView-3 dataset. For other dataset
# (i.e., QuickBird and GaoFen-2), please change the corresponding
# inputs.
# cd/d E:\LiuYu\lunwen\WSDFNet-main\codes

# ------------------------------------------------------------------

import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data4b import Dataset_Pro
import scipy.io as sio
from model4b import WSDFNet
import numpy as np
import shutil
from torch.utils.tensorboard import SummaryWriter


# ================== Pre-test =================== #
def load_set(file_path):
    data = sio.loadmat(file_path)  # HxWxC=256x256x8

    # tensor type:
    lms = torch.from_numpy(
        data['lms'] / 2047.0).permute(2, 0, 1)  # CxHxW = 8x256x256
    ms = torch.from_numpy((data['ms'] / 2047.0)
                          ).permute(2, 0, 1)  # CxHxW= 8x64x64
    pan = torch.from_numpy((data['pan'] / 2047.0))   # HxW = 256x256

    return lms, ms, pan



# ================== Pre-Define =================== #
SEED = 10
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
cudnn.deterministic = True

# ============= HYPER PARAMS(Pre-Defined) ==========#
lr = 0.0003
epochs = 500
ckpt = 50
batch_size = 32
device = torch.device('cuda:0')

model = WSDFNet().to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr,
                       betas=(0.9, 0.999))   # optimizer 1


if os.path.exists('train_logs'):  # for tensorboard: copy dir of train_logs
    # ---> console (see tensorboard): tensorboard --logdir = dir of train_logs
    shutil.rmtree('train_logs')
writer = SummaryWriter('train_logs')


def save_checkpoint(model, epoch, model_out_path):  # save model function
    # model_out_path = 'Weights/WSDFNET_{}.pth'.format(epoch)
    model_out_path = os.path.join(model_out_path,'WSDFNET_{}.pth').format(epoch)
    # if not os.path.exists(model_out_path):
    #     os.makedirs(model_out_path)
    torch.save(model.state_dict(), model_out_path)

# 定义一个整合的模型，其结构与你的传感器模型相同
class IntegratedModel(nn.Module):
    def __init__(self):
        super(IntegratedModel, self).__init__()
        self.sensor1 = WSDFNet()
        self.sensor2 = WSDFNet()
        self.sensor3 = WSDFNet()

    def forward(self, pan, lms):
        out_sensor1 = self.sensor1(pan, lms)
        out_sensor2 = self.sensor2(pan, lms)
        out_sensor3 = self.sensor3(pan, lms)
        return out_sensor1, out_sensor2, out_sensor3

# 创建整合模型
integrated_model = IntegratedModel().to(device)
# 加载预训练权重
integrated_model.sensor1.load_state_dict(torch.load('F:\AFusionGroup\Shiyan\shiyan20231201\IK_WSDFNet\SaveModel\WSDFNET_500.pth'))
integrated_model.sensor2.load_state_dict(torch.load('F:\AFusionGroup\Shiyan\shiyan20231201\QB_WSDFNet\SaveModel\WSDFNET_500.pth'))
integrated_model.sensor3.load_state_dict(torch.load('F:\AFusionGroup\Shiyan\shiyan20231201\WV3_WSDFNet\SaveModel\WSDFNET_500.pth'))

###################################################################
# ------------------- Main Train (Run second)----------------------
###################################################################

def train(training_data_loader, validate_data_loader, model_out_path):
    print('Start training...')

    for epoch in range(epochs):

        epoch += 1
        epoch_train_loss, epoch_val_loss = [], []

        # ============Epoch Train=============== #
        integrated_model.train()

        for iteration, batch in enumerate(training_data_loader, 1):
            gt, lms, _, _, pan = Variable(batch[0], requires_grad=False).to(device), \
                Variable(batch[1]).to(device), \
                batch[2], \
                batch[3], \
                Variable(batch[4]).to(device)
            # optimizer.zero_grad()  # fixed
            # out = model(pan, lms)

            # 使用整合模型进行前向传播
            out_sensor1, out_sensor2, out_sensor3 = integrated_model(pan, lms)

            # loss = criterion(out, gt)  # compute loss
            # 计算损失（根据实际情况修改）
            loss_sensor1 = criterion(out_sensor1, gt)
            loss_sensor2 = criterion(out_sensor2, gt)
            loss_sensor3 = criterion(out_sensor3, gt)
            total_loss = loss_sensor1 + loss_sensor2 + loss_sensor3

            # save all losses into a vector for one epoch
            # epoch_train_loss.append(loss.item())
            epoch_train_loss.append(total_loss.item())

            # 反向传播和优化（根据实际情况修改）
            # loss.backward()  # fixed
            total_loss.backward()
            # optimizer.step()  # fixed
            optimizer.step()

 #       lr_scheduler.step()  # update lr

        # compute the mean value of all losses, as one epoch loss
        t_loss = np.nanmean(np.array(epoch_train_loss))
        # write to tensorboard to check
        writer.add_scalar('train/loss', t_loss, epoch)
        # print loss for each epoch
        print('Epoch: {}/{} training loss: {:.7f}'.format(epochs, epoch, t_loss))

        if epoch % ckpt == 0:  # if each ckpt epochs, then start to save model
            # save_checkpoint(model, epoch)
            save_checkpoint(integrated_model, epoch, model_out_path)

        integrated_model.eval()
        with torch.no_grad():
            for iteration, batch in enumerate(validate_data_loader, 1):
                gt, lms, _, _, pan = Variable(batch[0], requires_grad=False).to(device), \
                    Variable(batch[1]).to(device), \
                    batch[2], \
                    batch[3], \
                    Variable(batch[4]).to(device)

                out = integrated_model(pan, lms)

                loss = criterion(out, gt)
                epoch_val_loss.append(loss.item())

        v_loss = np.nanmean(np.array(epoch_val_loss))
        writer.add_scalar('val/loss', v_loss, epoch)
        print('validate loss: {:.7f}'.format(v_loss))

    writer.close()  # close tensorboard


###################################################################
# ------------------- Main Function (Run first) -------------------
###################################################################

# conda activate LYpy3.8pt1.9.0
# cd/d C:\Users\LiuYu\Desktop\FusionEvaluateExperiment\DL_WSDFNet\codes
# python Runtrain4b2047qyxx.py

if __name__ == "__main__":
    
    train_set = Dataset_Pro('F:/AFusionGroup/Shiyan/shiyan20231201/A_Data/Train/N_C_H_W.mat')  
    training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True)  # put training data to DataLoader for batches
    validate_set = Dataset_Pro('F:/AFusionGroup/Shiyan/shiyan20231201/A_Data/Validation/N_C_H_W.mat')
    validate_data_loader = DataLoader(dataset=validate_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True)  # put training data to DataLoader for batches
    model_out_path = 'F:/AFusionGroup/Shiyan/shiyan20231201/A_WSDFNet/SaveModel_qyxx'
    if not os.path.exists(model_out_path):
        os.makedirs(model_out_path)
    train(training_data_loader, validate_data_loader, model_out_path)
