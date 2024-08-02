import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
import argparse

import config
from ctpn_model import CTPN_Model, RPN_CLS_Loss, RPN_REGR_Loss
from data.dataset import ICDARDataset

# dataset_download:https://rrc.cvc.uab.es/?ch=8&com=downloads
random_seed = 2019
torch.random.manual_seed(random_seed)   # 固定随机种子，用于初试化权重等操作，可保证实验的复现性
np.random.seed(random_seed)     # 可能需要同时使用torch和numpy，所以需要同时设置相同的随机种子

epochs = 30
lr = 1e-3
resume_epoch = 0


def save_checkpoint(state, epoch, loss_cls, loss_regr, loss, ext='pth'):
    """
    保存模型
    """
    check_path = os.path.join(config.checkpoints_dir,
                              f'v3_ctpn_ep{epoch:02d}_'
                              f'{loss_cls:.4f}_{loss_regr:.4f}_{loss:.4f}.{ext}')

    try:
        torch.save(state, check_path)
    except BaseException as e:
        print(e)
        print('fail to save to {}'.format(check_path))
    print('saving to {}'.format(check_path))

def weights_init(m):
    """
        初始化权重
        m.weight访问模块的权重参数
        .data：是tensor的属性，可以获取tensor的值，但最新版已不推荐使用，
                因为它直接访问了底层tensor的存储，而绕开了pytorch的自动求导机制
                现一般直接使用m.weight.normal_
        .normal_：正态分布初始化(均值，方差)，默认为(mean=0, std=1.0)
                下划线_表示这是一个原地操作，即直接修改了m.weight的值，可以显著提高内存效率
                因为(数据量大时)不需要拷贝副本，但要小心结构和边界，并具有不可逆性
    """
    classname = m.__class__.__name__    # 获取类名，用于根据对象类型动态地调用不同的方法
    if classname.find('Conv') != -1:    
        m.weight.data.normal_(0.0, 0.02)    
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)    # 初始化偏置为0


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    checkpoints_weight = config.pretrained_weights           # 预训练权重
    # print('exist pretrained ',os.path.exists(checkpoints_weight))
    if os.path.exists(checkpoints_weight):       # 存在预训练权重，则使用预训练权重
        pretrained = False

    dataset = ICDARDataset(config.icdar17_mlt_img_dir, config.icdar17_mlt_gt_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=config.num_workers)
    model = CTPN_Model()
    model.to(device)
    
    if os.path.exists(checkpoints_weight):
        print('using pretrained weight: {}'.format(checkpoints_weight))
        cc = torch.load(checkpoints_weight, map_location=device)
        model.load_state_dict(cc['model_state_dict'])
        resume_epoch = cc['epoch']
    else:
        model.apply(weights_init)

    params_to_uodate = model.parameters()
    optimizer = optim.SGD(params_to_uodate, lr=lr, momentum=0.9)
    
    critetion_cls = RPN_CLS_Loss(device)
    critetion_regr = RPN_REGR_Loss(device)
    
    best_loss_cls = 100
    best_loss_regr = 100
    best_loss = 100
    best_model = None
    epochs += resume_epoch
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)   # 学习率调度器，动态调整学习率
    
    for epoch in range(resume_epoch+1, epochs):
        print(f'Epoch {epoch}/{epochs}')
        print('#'*50)
        epoch_size = len(dataset) // 1  # 换成batch_size更好
        model.train()
        epoch_loss_cls = 0
        epoch_loss_regr = 0
        epoch_loss = 0
        scheduler.step(epoch)
    
        for batch_i, (imgs, clss, regrs) in enumerate(dataloader):
            # print(imgs.shape)
            imgs = imgs.to(device)
            clss = clss.to(device)
            regrs = regrs.to(device)
    
            optimizer.zero_grad()
    
            out_cls, out_regr = model(imgs)
            loss_cls = critetion_cls(out_cls, clss)
            loss_regr = critetion_regr(out_regr, regrs)
    
            loss = loss_cls + loss_regr  # total loss
            loss.backward()
            optimizer.step()
    
            epoch_loss_cls += loss_cls.item()
            epoch_loss_regr += loss_regr.item()
            epoch_loss += loss.item()
            mmp = batch_i+1
    
            print(f'Ep:{epoch}/{epochs-1}--'
                  f'Batch:{batch_i}/{epoch_size}\n'
                  f'batch: loss_cls:{loss_cls.item():.4f}--loss_regr:{loss_regr.item():.4f}--loss:{loss.item():.4f}\n'
                  f'Epoch: loss_cls:{epoch_loss_cls/mmp:.4f}--loss_regr:{epoch_loss_regr/mmp:.4f}--'
                  f'loss:{epoch_loss/mmp:.4f}\n')
    
        # scheduler.step(epoch)
        epoch_loss_cls /= epoch_size
        epoch_loss_regr /= epoch_size
        epoch_loss /= epoch_size
        print(f'Epoch:{epoch}--{epoch_loss_cls:.4f}--{epoch_loss_regr:.4f}--{epoch_loss:.4f}')
        if best_loss_cls > epoch_loss_cls or best_loss_regr > epoch_loss_regr or best_loss > epoch_loss:
            best_loss = epoch_loss
            best_loss_regr = epoch_loss_regr
            best_loss_cls = epoch_loss_cls
            best_model = model
            save_checkpoint({'model_state_dict': best_model.state_dict(),
                             'epoch': epoch},
                            epoch,
                            best_loss_cls,
                            best_loss_regr,
                            best_loss)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

