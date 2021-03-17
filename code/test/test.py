import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
from torchvision import transforms
import segmentation_models_pytorch as smp
import torch.nn.functional as F
import torch
import os  
from pathlib import  Path
import numpy as np
import torchvision.transforms as T
from pathlib import Path
import pandas as pd
import cv2
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
subm = []

data_path = Path("../tcdata/地表建筑物识别")
model_parh = Path("../user_data/model_data")
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

def load_model(filename, classes, channels, mode):
    if mode == 'b7':
        model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b7",
            encoder_weights=None,
            in_channels=channels,
            classes=classes)
    else:
        model = smp.UnetPlusPlus(
            encoder_name="timm-efficientnet-b8",
            encoder_weights=None,
            in_channels=channels,
            classes=classes,
            decoder_attention_type='scse')

    weights = torch.load(filename,map_location=torch.device('cpu'))
    model.load_state_dict(weights)
    
    return model
print("开始读入模型")
modelfile1 = model_parh/'stage-0_unet++_b8_80_attention_yaogan_best_model.pth'
modelfile2 = model_parh/'stage-0_unet++_b8_120_attention_yaogan_best_model.pth'

learn1 = load_model(modelfile1, 2, 3, 'b8')
learn2 = load_model(modelfile2, 2, 3, 'b8')
print("读入模型完成")

test_mask = pd.read_csv(data_path/'test_b_samplesubmit.csv', sep='\t', names=['name', 'mask'])
n_miou1 = [0.4999872738582193, 0.5021976002176444]
n_miou2 = [0.5000127261417807, 0.4978023997823557]

# 预测结果
pre_path = Path("../prediction_result")
if not pre_path.exists():
    pre_path.mkdir(parents=True, exist_ok=True)

print("开始预测结果")
path = data_path/"test_b"
for idx, name in enumerate(test_mask['name'].iloc[:]):
    img = transforms.ToTensor()(Image.open(path/name).resize((256,256), Image.BILINEAR)).unsqueeze(0)
    out1 = learn1.predict(img)
    out2 = learn2.predict(img)
    out = torch.zeros_like(out1)
    for i in range(2):
        out[:,i,:,:] = out1[:,i,:,:]*n_miou1[i]+out2[:,i,:,:]*n_miou2[i]

    res = out.squeeze(0).detach().numpy()
    res = np.argmax(res,axis=0).reshape((256,256)).astype(np.uint8)
    tar = cv2.resize(res, (512,512))
    subm.append([name, rle_encode(tar)])
print("写入文件")
subm = pd.DataFrame(subm)
subm.to_csv(pre_path/'result.csv', index=None, header=None, sep='\t')
print("预测完成")
