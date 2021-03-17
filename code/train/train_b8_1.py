import numpy as np
import pandas as pd
from pathlib import Path
import torch.nn as nn
from PIL import Image
from fastai.data import *
from fastai.torch_core import *
from fastai.vision.all import *
from fastai.callback import *
import copy
from loguru import logger
import os
import torch
import segmentation_models_pytorch as smp
from fastai.vision.core import PILBase
#PILBase._open_args = {'mode':'RGBA'}
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

path = Path("../user_data/tmp_data")
model_parh = Path("../user_data/model_data")
data_name = 'jianzhu'
default_lr = 1e-3
default_epochs = 5

train_params = {
    'train_size' : 256,
    'train_bs' : 10,
    'num_class' : 2,
    'one_cycle_policy' : False,
    'opt_func' : 'Adam',
    'loss_func' : None,
    'metrics' : '[acc_image_seg, miou_image_seg]',
    'wd' : 1e-2,
    'get_transform' : {'flip_vert': True},
    'fit_policy' : {
       'stage-0' : {'freeze_to':0, 'lr':'%s' % default_lr, 'epochs':80, 'wd':None},
       # 'stage-1' : {'freeze_to':2, 'lr':'slice(%s/10, %s)' % (default_lr, default_lr), 'epochs':10, 'wd':None},
       # 'stage-2' : {'freeze_to':1, 'lr':'%s/2.' % default_lr, 'epochs':20, 'wd':None},
       # 'stage-3' : {'freeze_to':1, 'lr':'slice(%s/2./10, %s/2.)' % (default_lr, default_lr), 'epochs':20, 'wd':None},
       # 'stage-4' : {'freeze_to':0, 'lr':'%s' % default_lr, 'epochs':100, 'wd':None},
       # 'stage-5' : {'freeze_to':0, 'lr':'slice(%s/3./10, %s/3.)' % (default_lr, default_lr), 'epochs':100, 'wd':None},
    },
    'custom_head' : None,
    'callbacks' : {
        #'EarlyStoppingCallback':{
        #    'monitor':'valid_loss',
        #    'patience':15,
        #},
        #'ReduceLROnPlateau':{
        #    'monitor':'valid_loss',
        #    'patience':10,
        #    'min_delta':0.1,
        #    'min_lr':0,
        #},
        'SaveModelCallback':{
            'monitor':'miou_image_seg',
            'fname':'unet++_b8_80_attention_yaogan_best_model',
        }
    }
}

def pixel_acc(pred, target):
    correct = (pred == target).sum()
    total   = (target == target).sum()
    return correct / total

def iou(pred, target):
    ious = []
    for cls in range(train_params.get('num_class')):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / max(union, 1))
    return ious

def acc_image_seg(output, Y):
    """ accuracy measure pixels """
    pixel_accs = []
    output = output.data.cpu().numpy()

    N, _, h, w = output.shape
    pred = output.transpose(0, 2, 3, 1).reshape(-1, train_params.get('num_class')).argmax(axis=1).reshape(N, h, w)

    target = Y.cpu().numpy().reshape(N, h, w)
    for p, t in zip(pred, target):
        pixel_accs.append(pixel_acc(p, t))
    return np.array(pixel_accs).mean()


def miou_image_seg(output, Y):
    """ miou """
    total_ious = []
    output = output.data.cpu().numpy()

    N, _, h, w = output.shape
    pred = output.transpose(0, 2, 3, 1).reshape(-1, train_params.get('num_class')).argmax(axis=1).reshape(N, h, w)

    target = Y.cpu().numpy().reshape(N, h, w)
    for p, t in zip(pred, target):
        total_ious.append(iou(p, t))

    total_ious = np.array(total_ious).T
    ious = np.nanmean(total_ious, axis=1)
    return np.nanmean(ious)

if __name__ == '__main__':
    model = smp.UnetPlusPlus(
        encoder_name="timm-efficientnet-b8",
        encoder_weights="imagenet",
        in_channels=3,
        classes=train_params.get('num_class'),
        decoder_attention_type='scse'
    )
    
    #conv1 = list(model.state_dict().keys())[0]
    #x = model.state_dict()[conv1]
    #x[:,-1,:,:] = x[:,0,:,:]

    #model.cuda()
    #model = torch.nn.DataParallel(model)

    # dls = HandelData(image_root=train_image_root, label_root=label_image_root, code_txt= code_txt_root)
    fnames = get_image_files(path/'image')
    def label_func(x): return path/'label'/f'{x.stem}.png'
    if train_params.get('get_transfrom') is not None:
        tfms = aug_transforms(**train_params.get('get_transform'))
    else:
        tfms = None
    dls = SegmentationDataLoaders.from_label_func(path, fnames, label_func, valid_pct=0.2,item_tfms = Resize(train_params.get('train_size'), method=ResizeMethod.Squish), 
    batch_tfms = tfms, 
    codes=np.loadtxt(path/'codes.txt', dtype=str), bs=train_params.get('train_bs'))

    learn = Learner(
        dls,
        model,
        loss_func = eval(train_params.get('loss_func')) if train_params.get('loss_func') else None,
        opt_func = partial(
            eval(train_params.get('opt_func')),
            lr=default_lr
        ),
        metrics = eval(train_params.get('metrics')) if train_params.get('metrics') else None,
        wd = train_params.get('wd'),
    )
    #learn.model = torch.nn.DataParallel(learn.model)
    learn = learn.to_fp16()

    #start train
    last_beat_mode_name = None
    for i, item in enumerate(train_params.get('fit_policy').items()):
        stage_name, one_fit_policy = item
        current_epochs = one_fit_policy.get('epochs')
        current_lr = one_fit_policy.get('lr')
        current_freeze_group = one_fit_policy.get('freeze_to')
        current_wd = one_fit_policy.get('wd')
        logger.info("freeze_to layer group %d, lr %s, train %d epochs" % (current_freeze_group, current_lr, current_epochs))
        learn.freeze_to(current_freeze_group)
        callback_dict = train_params.get('callbacks')
        callbacks = None
        if callback_dict is not None:
            callbacks = []
            for k, v in callback_dict.items():
                kwargs = copy.deepcopy(v)
                if 'fname' in v:
                    kwargs['fname'] = "_".join([stage_name, v['fname']])
                    if k == 'SaveModelCallback':
                        learn.save(kwargs['fname'])
                    last_beat_mode_name = kwargs['fname']
                callbacks.append(eval(k)(**kwargs))
        if train_params.get('one_cycle_policy'):
            learn.fit_one_cycle(current_epochs, eval(current_lr), wd=current_wd, cbs=callbacks)
        else:
            learn.fit_flat_cos(current_epochs, eval(current_lr), wd=current_wd, cbs=callbacks)
        if last_beat_mode_name is not None:
            learn.load(last_beat_mode_name)
            logger.info("beat mode %s" % last_beat_mode_name)
    
    shutil.copy(path/'models'/'stage-0_unet++_b8_80_attention_yaogan_best_model.pth', model_parh/'stage-0_unet++_b8_80_attention_yaogan_best_model.pth')
