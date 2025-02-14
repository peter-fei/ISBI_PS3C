import os
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping
from torch.utils.data import DataLoader,Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch
from torch import nn
import timm
from torchmetrics import MetricCollection, Recall, Specificity, AUROC, Precision, F1Score, Accuracy,ConfusionMatrix
from torchmetrics.classification import BinaryRecall,BinarySpecificity
import argparse
from torchvision.models import resnet50
import json
from torch.nn.functional import one_hot, log_softmax
from albumentations import (
    Compose,
    GaussianBlur,
    HorizontalFlip,
    MedianBlur,
    MotionBlur,
    Normalize,
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
import os
from sklearn.model_selection import KFold
from dataset_isbi import AlbumentationsDataset,InferenceDataset
import numpy as np
import functools
import torch.optim.lr_scheduler as lr_scheduler
import math
import pandas as pd
from tqdm import tqdm
from pytorch_lightning import seed_everything
from lora import LoRA_ViT_timm

seed_everything(42)

os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"
parser = argparse.ArgumentParser()
parser.add_argument('--encoder',default='')
parser.add_argument('--batch_size',default=128,type=int)
parser.add_argument('--lr',default=1e-4,type=float)
parser.add_argument('--save_dir',default='lora')
parser.add_argument('--epochs',default=50,type=int)


args = parser.parse_args()
img_size = 224

def load_cfg_from_json(json_file):
    with open(json_file, "r", encoding="utf-8") as reader:
        text = reader.read()
    return json.loads(text)


if args.encoder == 'uni':
    encoder_ori = timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True)
    encoder_ori.load_state_dict(torch.load("checkpoints/UNI/pytorch_model.bin", map_location="cpu"), strict=True)
    in_feature = 1024
    print('uni---------------------------------')
elif args.encoder == 'gigapath':
    cfg = load_cfg_from_json("checkpoints/prov-gigapath/config.json")
    encoder_ori = timm.create_model("vit_giant_patch14_dinov2",**cfg['model_args'],pretrained_cfg=cfg['pretrained_cfg'], dynamic_img_size=True, pretrained=False,checkpoint_path="checkpoints/prov-gigapath/pytorch_model.bin")
    print(encoder_ori.load_state_dict(torch.load('checkpoints/prov-gigapath/pytorch_model.bin'),strict=False),'gigapath_weights')
    in_feature = 1536
    print('gigapath---------------------------------')
elif args.encoder == 'hoptimus':
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
    encoder_ori = timm.models.VisionTransformer(**params)
    encoder_ori.load_state_dict(torch.load('checkpoints/Hoptimus/checkpoint.pth', map_location="cpu"))
    in_feature = 1536


img_size = 224
print(args.encoder,img_size,'********************************************************')

train_transform = Compose(
    [

        LongestMaxSize(max_size=224, interpolation=cv2.INTER_LINEAR, always_apply=True),
        PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), p=1.0),
        RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        VerticalFlip(p=0.5),
        HorizontalFlip(p=0.5),
        ShiftScaleRotate(
            shift_limit=0.2,
            scale_limit=0.2,
            rotate_limit=20,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            p=1,
        ),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ]
)

val_transform = Compose([

    LongestMaxSize(max_size=224, interpolation=cv2.INTER_LINEAR, always_apply=True),  # 等比例缩放，最长边为224
    PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), p=1.0),
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


class LoRA(LightningModule):
    def __init__(self, in_feature=768, hidden=768, num_classes=2, lr=0.0001,epochs=30,ft=True,encoder=None):
        super().__init__()
        # self.save_hyperparameters()
        self.mlp = nn.Sequential(
            nn.Linear(in_feature, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, num_classes)
        )
        if encoder is None:
            self.save_hyperparameters()
            encoder = encoder_ori
        self.encoder = LoRA_ViT_timm(vit_model=encoder, r=4, alpha=4, num_classes=num_classes)


        criterion = PolyLoss(num_classes=num_classes)

        self.criterions = [criterion]


        print(num_classes,'num_class')

        metrics = MetricCollection([
            Accuracy(task='multiclass' if num_classes>2 else 'binary',num_classes=num_classes,average='micro'),
            Recall(task='multiclass' if num_classes>2 else 'binary',num_classes=num_classes,average='macro'),
            Specificity(task='multiclass' if num_classes>2 else 'binary',num_classes=num_classes,average='macro'),
            Precision(task='multiclass' if num_classes>2 else 'binary',num_classes=num_classes,average='macro'),
            F1Score(task='multiclass' if num_classes>2 else 'binary',num_classes=num_classes,average='macro'),
        ])

        top_k_metrics = MetricCollection({
            'top_1_acc': Accuracy(top_k=1,task='multiclass',num_classes=num_classes),
            'top_2_acc': Accuracy(top_k=2,task='multiclass',num_classes=num_classes),
            'top_1_sen': Recall(top_k=1,task='multiclass' if num_classes>2 else 'binary',num_classes=num_classes,average='micro'),
            'top_2_sen': Recall(top_k=2,task='multiclass' if num_classes>2 else 'binary',num_classes=num_classes,average='micro'),
            'top_1_f1': F1Score(top_k=1,task='multiclass',num_classes=num_classes,average='micro'),
            'top_2_f1': F1Score(top_k=2,task='multiclass',num_classes=num_classes,average='micro'),         
        })

        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')      

        self.test_top_k_metrics = top_k_metrics.clone(prefix='test_')

        
        self.test_confusion = MetricCollection([ConfusionMatrix(task="multiclass", num_classes=num_classes)])

        
        self.test_acc =  Accuracy(task='multiclass' if num_classes>2 else 'binary',num_classes=num_classes,average='none')
        self.test_sen = Recall(task='multiclass' if num_classes>2 else 'binary',num_classes=num_classes,average='none')
        self.test_spec = Specificity(task='multiclass' if num_classes>2 else 'binary',num_classes=num_classes,average='none')
        self.test_auc = AUROC(task='multiclass' if num_classes>1 else 'binary',num_classes=num_classes,average='none')
        self.test_f1 = F1Score(task='multiclass' if num_classes>2 else 'binary',num_classes=num_classes,average='none')
        
        self.val_acc = Accuracy(task='multiclass' if num_classes>2 else 'binary',num_classes=num_classes,average='none')
        self.val_f1 = F1Score(task='multiclass' if num_classes>2 else 'binary',num_classes=num_classes,average='none')

        self.val_f1_weighted = F1Score(task='multiclass' if num_classes>2 else 'binary',num_classes=num_classes,average='weighted')




    def forward(self, x):
        out = self.encoder(x)
        return out
        
        

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = 0
        for loss_fn in self.criterions:
            loss += loss_fn(logits, y)
        
        self.log('train_loss', loss,on_step=True,on_epoch=True, prog_bar=True)
        preds = torch.max(logits, dim=1)[1]
        self.train_metrics.update(preds, y)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = 0
        for loss_fn in self.criterions:
            loss += loss_fn(logits, y)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        preds = torch.max(logits, dim=1)[1]
        self.val_metrics.update(preds, y)

        self.log_dict(self.val_metrics, on_step=False, on_epoch=True)
        self.val_acc.update(preds, y)
        self.val_f1.update(preds, y)

        self.val_f1_weighted.update(preds, y)
        self.log('val_f1_weighted', self.val_f1_weighted, on_step=False, on_epoch=True, prog_bar=True)

        self.log_dict(self.val_metrics, on_step=False, on_epoch=True)
        
        return loss



    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        self.log('test_loss', loss)

        preds = torch.max(logits, dim=1)[1]

        self.test_metrics.update(preds, y)
        self.test_confusion.update(preds, y)
        self.test_auc.update(torch.softmax(logits,1),y)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True)
        self.log_dict(self.test_top_k_metrics,on_step=False, on_epoch=True)
        self.test_sen.update(preds,y)
        self.test_spec.update(preds,y)
        self.test_acc.update(preds,y)
        self.test_f1.update(preds,y)

        return loss
    

    def on_train_epoch_end(self):
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        self.val_metrics.reset()
        print()
        print('ACC  ',self.val_acc.compute())
        print('F1   ',self.val_f1.compute())
        print('F1-Weighted   ',self.val_f1_weighted.compute())

        self.val_acc.reset()
        self.val_f1.reset()
        self.val_f1_weighted.reset()
    
    def on_test_epoch_end(self):

        confusion = self.test_confusion.compute()
        print('confusion_matrix')
        print(confusion)

        sen = self.test_sen.compute()
        spec = self.test_spec.compute()
        f1 = self.test_f1.compute()
        
        auc = self.test_auc.compute()
        acc = self.test_acc.compute()
        print('acc',acc)
        print('auc',auc)
        print('sen',sen)
        print('spec',spec)
        print('f1',f1)

        self.test_metrics.reset()
        self.test_sen.reset()
        self.test_acc.reset()
        self.test_spec.reset()
        self.test_f1.reset()
        self.test_auc.reset()
        self.test_confusion.reset()

    def configure_optimizers(self):
        print('lr',self.hparams.lr)
        lrf = 0.1
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr) 
        lf = lambda x: ((1 + math.cos(x * math.pi / self.hparams.epochs)) / 2) * (1 - lrf) + lrf
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95, last_epoch=-1, verbose='deprecated')
        op = {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',  # 调度器的更新频率，可以是 'step' 或 'epoch'
                'frequency': 1,       # 调度器的更新频率
                'name': 'lambda_lr'   # 调度器的名称
            }
        }
        return op
    




def train_ISBI():
    dataset = AlbumentationsDataset('Datas/ISBI2025/isbi2025-ps3c-train-dataset/',transform=train_transform)
    folds = KFold(n_splits=10, shuffle=True, random_state=2024)
    datas = np.array([sample[0] for sample in dataset.samples])
    labels = np.array([sample[1] for sample in dataset.samples])

    num_classes = 3
    print(type(dataset[0]))
    print(len(dataset),num_classes,dataset.classes)

    for fold_i, (train_index, val_index) in enumerate(folds.split(datas,labels)):

        print(fold_i,'----------------------------------------------------------')
        # # fold_i = fold_i
        train_dataset = Subset(dataset, train_index)
        print(train_index,len(train_index))
        print(val_index,len(val_index))
        val_dataset = AlbumentationsDataset('Datas/ISBI2025/isbi2025-ps3c-train-dataset/', transform=val_transform)
     
        val_dataset.samples = [dataset.samples[i] for i in val_index]

        nw = 4
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=nw)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,num_workers=nw)
        # test_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,num_workers=nw)

        checkpoint_callback = ModelCheckpoint(
            monitor='val_f1_weighted',
            dirpath=f'ckpts/ISBI2025/{args.save_dir}',
            filename='mlp-{epoch:02d}-{val_f1_weighted:.3f}',
            save_top_k=3,
            mode='max',
            save_last=True
        )
        early_stop_callback = EarlyStopping(
            monitor='val_f1_weighted',  
            min_delta=0.00,     
            patience=40,         
            verbose=True,        
            mode='max'       
        )
        trainer = Trainer(
            max_epochs=args.epochs,
            callbacks=[checkpoint_callback, early_stop_callback],
            log_every_n_steps=1
        )
        
        args.ft = True
        
        model = LoRA(in_feature=in_feature, hidden=768, num_classes=num_classes,lr=args.lr,epochs=args.epochs,ft=args.ft)
        trainer.fit(model, train_loader, val_loader)
        model.eval()
        exit()

@torch.no_grad()
def infer_ISBI():
    nw = 4
    dataset = InferenceDataset('Datas/ISBI2025/isbi2025-ps3c-test-dataset/',val_transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,num_workers=nw)
    model = LoRA.load_from_checkpoint('ckpts/ISBI2025/0210/uni_lora_poly/mlp.ckpt')
    model.eval()
    model.cuda()
    label_dict = {0:'unhealthy',1:'healthy',2:'rubbish',3:'unhealthy'}
    preds = []
    pred_names = []
    labels = []
    for sample in tqdm(loader):
        x,names = sample
        x = x.cuda()
        # print(x.size())
        res = model(x)
        pred = torch.max(res,dim=-1)[1]
        for i in range(len(names)):
            preds.append({'image_name':names[i],'label':label_dict[pred[i].item()]})
            pred_names.append(names[i])
            labels.append(label_dict[pred[i].item()])
            # print({'image_name':names[i],'label':label_dict[pred[i].item()]})
        # exit()
    df = pd.DataFrame({'image_name':pred_names,'label':labels})
    df.to_csv('test.csv',index=False)
    df = pd.DataFrame(preds)
    df.to_csv('0107_noboth_lora.csv',index=False)


class PolyLoss(torch.nn.Module):
    """
    Implementation of poly loss.
    Refers to `PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions (ICLR 2022) <https://arxiv.org/abs/2204.12511>`_.
    """
    def __init__(self, num_classes=1000, epsilon=1.0,weights=None):
        super().__init__()
        self.epsilon = epsilon
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.num_classes = num_classes
        self.weights = weights

    def forward(self, output, target):
        ce = self.criterion(output, target)
        pt = one_hot(target, num_classes=self.num_classes) * self.log_softmax(output)
        if self.weights is not None:
            self.weights = self.weights.to(pt.device)
            ce *= self.weights[target]
        return (ce + self.epsilon * (1.0 - pt.sum(dim=-1))).mean()

if __name__ == '__main__':
    train_ISBI()
    # infer_ISBI()

    

        

