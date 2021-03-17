import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from PIL import Image
import math
import shutil
from fastai.torch_core import parallel
import random

data_path = Path("../tcdata/地表建筑物识别")
train_data_path = Path("../user_data/tmp_data")

test_img = train_data_path/"test"/"images"
test_lab = train_data_path/"test"/"labels"

if not test_img.exists():
    test_img.mkdir(parents=True, exist_ok=True)
if not test_lab.exists():
    test_lab.mkdir(parents=True, exist_ok=True)

# 将图片编码为rle格式
def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten(order = 'F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# 将rle格式进行解码为图片
def rle_decode(mask_rle, shape=(512, 512)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')

label_path = train_data_path/"label"
if not label_path.exists():
    label_path.mkdir(parents=True, exist_ok=True)
train_mask = pd.read_csv(data_path/'train_mask.csv', sep='\t', names=['name', 'mask'])
for i in range(len(train_mask)):
    if str(train_mask['mask'].iloc[i]) != 'nan':
        label_name = train_mask['name'].iloc[i][:-4]+'.png'
        mask = rle_decode(train_mask['mask'].iloc[i])
        lab = Image.fromarray(mask)
        lab.save(label_path/label_name, quality=100)

def cp(i):
    img_path = data_path/"train"
    save_path = train_data_path/"image"
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)
    shutil.copy(img_path/(i.stem+'.jpg'), save_path/(i.stem+'.jpg'))

parallel(cp, [i for i in (train_data_path/"label").rglob("*.png")])

print("开始提出1000张测试集")
a = [i for i in (train_data_path/'image').rglob("*.jpg")]
b = random.sample(a, 1000)
for i in b:
    shutil.move(i, test_img/i.name) 
    shutil.move(train_data_path/"label"/(i.stem+'.png'), test_lab/(i.stem+'.png'))
print("提出完成")