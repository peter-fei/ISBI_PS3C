
from torch.utils.data import DataLoader
import torch
import json
from albumentations import (
    Compose,
    GaussianBlur,
    HorizontalFlip,
    MedianBlur,
    MotionBlur,
    Normalize,
    OneOf,
    RandomBrightnessContrast,
    Resize,
    ShiftScaleRotate,
    VerticalFlip,
    CenterCrop,
    PadIfNeeded,
    LongestMaxSize
)
from albumentations.pytorch import ToTensorV2
import cv2
from dataset_isbi import InferenceDataset
import functools
import pandas as pd
from tqdm import tqdm
import re
from autogluon.tabular import TabularPredictor
from autogluon.tabular import TabularDataset
from main import LoRA
import timm


nw = 8
img_size = 224
val_transform = Compose([
    LongestMaxSize(max_size=224, interpolation=cv2.INTER_LINEAR, always_apply=True),
    PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), p=1.0),
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


def load_cfg_from_json(json_file):
    with open(json_file, "r", encoding="utf-8") as reader:
        text = reader.read()
    return json.loads(text)



def insert_number(s, num=1):
    # 如果没有下划线，直接在末尾加上数字
    if '_' not in s:
        return f"{s}_{num}"
    # 否则，替换第一个下划线前插入数字
    return re.sub(r'(.*?)_', fr'\1_{num}_', s, count=1)

@torch.no_grad()
def infer_ISBI_test():
    uni_model = timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True)
    uni_model.load_state_dict(torch.load("/public/home/jianght2023/checkpoints/UNI/pytorch_model.bin", map_location="cpu"), strict=True)

    cfg = load_cfg_from_json("/public/home/jianght2023/checkpoints/prov-gigapath/config.json")
    giga_model = timm.create_model("vit_giant_patch14_dinov2",**cfg['model_args'],pretrained_cfg=cfg['pretrained_cfg'], dynamic_img_size=True, pretrained=False,checkpoint_path="/public/home/jianght2023/checkpoints/prov-gigapath/pytorch_model.bin")
    print(giga_model.load_state_dict(torch.load('/public/home/jianght2023/checkpoints/prov-gigapath/pytorch_model.bin'),strict=False),'gigapath_weights')

    params = {
        'patch_size': 14, 
        'embed_dim': 1536, 
        'depth': 40, 
        'num_heads': 24, 
        'init_values': 1e-05, 
        'mlp_ratio': 5.33334, 
        'mlp_layer': functools.partial(
            timm.layers.mlp.GluMlp, act_layer=torch.nn.modules.activation.SiLU, gate_last=False
        ), 
        'act_layer': torch.nn.modules.activation.SiLU, 
        'reg_tokens': 4, 
        'no_embed_class': True, 
        'img_size': 224, 
        'num_classes': 0, 
        'in_chans': 3,
        'dynamic_img_size' : True,
    }
    hop_model = timm.models.VisionTransformer(**params)
    hop_model.load_state_dict(torch.load('checkpoints/Hoptimus/checkpoint.pth', map_location="cpu"))


    dataset = InferenceDataset('Datas/ISBI2025/ps-3-c-final-evaluation/',val_transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=False,num_workers=nw)
    
    uni = LoRA.load_from_checkpoint('ckpts/ISBI2025/uni_lora/mlp.ckpt',encoder=uni_model)
    hoptimus = LoRA.load_from_checkpoint('ckpts/ISBI2025/hoptimus_lora/mlp.ckpt',encoder=hop_model)
    gigapath = LoRA.load_from_checkpoint('ckpts/ISBI2025/giga_lora/mlp.ckpt',encoder=giga_model)

    uni.eval()
    uni.cuda()
    hoptimus.eval()
    hoptimus.cuda()
    gigapath.eval()
    gigapath.cuda()

    encoder_columns = ['gigapath','hoptimus','uni']
    models = [gigapath,hoptimus,uni]

    preds = []

    for sample in tqdm(loader):
        x,names = sample
        x = x.cuda()

        items = [{'image_name':names[i] } for i in range(len(names))]
        for model,name_column in zip(models,encoder_columns):
            model.cuda()
            res = model(x)
            pred = torch.softmax(res,dim=-1)
            for i in range(len(names)):
                
                items[i][insert_number(name_column, 0)] = pred[i,0].item()
                items[i][insert_number(name_column, 1)] = pred[i,1].item()
                items[i][insert_number(name_column, 2)] = pred[i,2].item()
        preds.extend(items)

    df = pd.DataFrame(preds)
    df.to_csv(f'./pred.csv',index=False)

def autogluon_infer():
    test_data = TabularDataset('pred.csv')
    loaded_predictor = TabularPredictor.load('AutogluonModels/ag-3/')
    predictions = loaded_predictor.predict(test_data)
    label_mapping = {0:'unhealthy',1:'healthy',2:'rubbish',3:'unhealthy'}

    predictions_mapped = predictions.map(label_mapping)

    # 创建包含 image_name 和预测结果的 DataFrame
    result_df = pd.DataFrame({
        'image_name': test_data['image_name'],  # 假设 val_data 中有 'image_name' 列
        'label': predictions_mapped
    })

    # 保存到 CSV 文件
    result_df.to_csv('pred_final.csv', index=False)

if __name__ == '__main__':
    infer_ISBI_test()
    autogluon_infer()
