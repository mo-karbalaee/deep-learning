import torch as t
from data import ChallengeDataset
from trainer import Trainer
import model
import pandas as pd
import os
import glob
import shutil
from sklearn.model_selection import KFold

os.makedirs('checkpoints', exist_ok=True)

data = pd.read_csv('data.csv', sep=';').reset_index(drop=True)

builders = [model.ResNet, model.build_resnet101, model.build_resnext50]
n_splits = len(builders)
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

fold_ckpts = []
for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
    print('===== fold {}/{} ====='.format(fold + 1, n_splits))
    train_data = data.iloc[train_idx]
    val_data = data.iloc[val_idx]

    train_labels = train_data[['crack', 'inactive']].values.astype(float)
    pos_counts = train_labels.sum(axis=0)
    class_weights = len(train_labels) / (pos_counts + 1.0)
    sample_weights = (train_labels * class_weights).sum(axis=1) + 1.0
    sampler = t.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_dl = t.utils.data.DataLoader(ChallengeDataset(train_data, 'train'), batch_size=64, sampler=sampler)
    val_dl = t.utils.data.DataLoader(ChallengeDataset(val_data, 'val'), batch_size=64, shuffle=False)

    res_model = builders[fold]()
    crit = t.nn.BCELoss()
    optim = t.optim.Adam(res_model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=3)
    trainer = Trainer(res_model, crit, optim, train_dl, val_dl,
                      cuda=t.cuda.is_available(), early_stopping_patience=10, scheduler=scheduler)
    trainer.fit(epochs=50)

    ckpts = sorted(glob.glob('checkpoints/checkpoint_*.ckp'))
    dst = 'checkpoints/fold_{}.ckp'.format(fold)
    shutil.copy(ckpts[-1], dst)
    fold_ckpts.append(dst)
    for c in ckpts:
        os.remove(c)

members = []
for fold, ck in enumerate(fold_ckpts):
    m = builders[fold]()
    m.load_state_dict(t.load(ck, map_location='cpu')['state_dict'])
    members.append(m)

ensemble = model.Ensemble(members)
export_trainer = Trainer(ensemble, t.nn.BCELoss(), cuda=False)
export_trainer.save_onnx('ensemble.onnx')
print('Saved ensemble.onnx from {} folds'.format(len(members)))
