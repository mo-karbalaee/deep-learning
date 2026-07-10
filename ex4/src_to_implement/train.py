import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
import os
from sklearn.model_selection import train_test_split

os.makedirs('checkpoints', exist_ok=True)

data = pd.read_csv('data.csv', sep=';')
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

train_labels = train_data[['crack', 'inactive']].values.astype(float)
pos_counts = train_labels.sum(axis=0)
class_weights = len(train_labels) / (pos_counts + 1.0)
sample_weights = (train_labels * class_weights).sum(axis=1) + 1.0
sampler = t.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_dl = t.utils.data.DataLoader(ChallengeDataset(train_data, 'train'), batch_size=64, sampler=sampler)
val_dl = t.utils.data.DataLoader(ChallengeDataset(val_data, 'val'), batch_size=64, shuffle=False)

res_model = model.ResNet()

crit = t.nn.BCELoss()
optim = t.optim.Adam(res_model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=3)
trainer = Trainer(res_model, crit, optim, train_dl, val_dl,
                  cuda=t.cuda.is_available(), early_stopping_patience=10, scheduler=scheduler)

res = trainer.fit(epochs=50)

plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')
