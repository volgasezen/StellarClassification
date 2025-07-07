# %%
os.chdir('/home/oban/Desktop/Volga/stellar-classification')
import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import (Dataset, TensorDataset, DataLoader, 
                              WeightedRandomSampler, random_split)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# %%
hdul = fits.open('data/dataset3_subset2.fits')
hdul1 = fits.open('data/dataset_subset4.fits')
dataset = hdul[1].data
dataset_old = hdul1[1].data
hdul.close()
hdul1.close()

problem = np.where((dataset_old.TYPED_ID == "HD134439") | 
                   (dataset_old.TYPED_ID == "Ross  889"))
dataset_old = np.delete(dataset_old, problem)
# %%
# trsvchn's answer on stackoverflow at: https://stackoverflow.com/questions/55588201/pytorch-transforms-on-tensordataset

class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)

class GaussianNoise(torch.nn.Module):
    def __init__(self, mean: float, sigma_r: tuple):
        super().__init__()
        self.mean = mean
        self.sigma_r = sigma_r

    def forward(self, spectra):
        sigma = torch.Tensor((1)).uniform_(*self.sigma_r).cuda()
        noise = self.mean + torch.randn_like(spectra).cuda() * sigma
        out = spectra + noise
        return out

def split_dataset(data, label, dataset, split, **kwargs):
    x = data[split]
    y = label[split]
    return dataset((x,y),**kwargs)

# %%
flux_numpy = dataset.FLUX.newbyteorder().byteswap()

from eval_utils import label_field, stellar_metrics

label_f = label_field(dataset, regr=False, new=True)
# label_f_old = label_field(dataset_old, regr=False, new=False)

classes = label_f.classes
indices = label_f.ord_to_idx(classes)

splits = list(range(0,len(dataset)))

train, test_val = train_test_split(splits, test_size=0.3, 
                                   random_state=1337, stratify=indices)

ones = np.where(np.unique(classes[test_val],return_counts=True)[1] == 1)

if len(ones) != 0: 
    for i in ones[0]:
        label = label_f.idx_to_ord(i)
        idx = np.where(classes[train] == label)[0][-1]
        real_idx = train.pop(idx)
        test_val.append(real_idx)

val, test = train_test_split(test_val, test_size=0.5, 
                             random_state=1337, stratify=indices[test_val])

assert sum([len(set(classes[i])) for i in [train, val, test]]) % 3 == 0
# %%
flux = torch.tensor(flux_numpy).cuda().float()
labels = torch.tensor(indices).cuda().int() #indices

gaussian_noise = GaussianNoise(mean=0,sigma_r=(0.01,0.05))

train_iter = split_dataset(flux, labels, CustomTensorDataset, 
                           train, transform=gaussian_noise)

val_iter = split_dataset(flux, labels, CustomTensorDataset, val)

test_iter = split_dataset(flux, labels, CustomTensorDataset, test)

weights = pd.Series(indices[train]).value_counts()

random_sample = WeightedRandomSampler(weights=max(weights)/weights[indices[train]].values,
                                      num_samples=len(indices[train]))

train_loader = DataLoader(train_iter, 128, sampler=random_sample)
val_loader = DataLoader(val_iter, 16)
test_loader = DataLoader(test_iter, 16)
# %%
def trainer(dataloader, model, loss, optimizer):
    losses = []
    preds = []
    actuals = []
    model.train()
    for fluxes, labels in dataloader:

        labels = labels.type(torch.LongTensor).cuda()
        fluxes = fluxes.cuda()

        optimizer.zero_grad()
        class_scores = model(fluxes)
        train_loss = loss(class_scores, labels)
        train_loss.backward()
        optimizer.step()
    
        losses.append(train_loss.item())
        preds.extend(class_scores.argmax(dim=1).tolist())
        actuals.extend(labels.tolist())

    inter_sm = stellar_metrics(label_f, preds, actuals, True, False)
    f1 = inter_sm.f1_macro()
    qwk = inter_sm.two_stage_qwk('q')

    return losses, f1, qwk

def validator(dataloader, model, loss, test=False):
    losses = []
    preds = []
    actuals = []
    model.eval()
    with torch.no_grad():
        if not test:
            for fluxes, labels in dataloader:
                labels = labels.type(torch.LongTensor).cuda()
                fluxes = fluxes.cuda()

                class_scores = model(fluxes)
                val_loss = loss(class_scores, labels)
                losses.append(val_loss.item())
                preds.extend(class_scores.argmax(dim=1).tolist())
                actuals.extend(labels.tolist())

            inter_sm = stellar_metrics(label_f, preds, actuals, True, False)
            f1 = inter_sm.f1_macro()
            qwk = inter_sm.two_stage_qwk('q')

            return losses, f1, qwk

        if test:
            for fluxes, labels in dataloader:
                labels = labels.type(torch.LongTensor).cuda()
                fluxes = fluxes.cuda()

                class_scores = model(fluxes)
                class_preds = class_scores.argmax(dim=1)
                preds.extend(class_preds.tolist())
                actuals.extend(labels.tolist())

            return preds, actuals

class2coords = torch.tensor(
    label_f.to_regr(np.unique(classes),idx=False),
    dtype=torch.float32)

class OrdinalLoss(nn.Module):
    def __init__(self, class2coords, alpha, betas=[1,1]):
        super().__init__()
        # register as a buffer so it moves with .to(device), but is not trained
        self.register_buffer('coords', class2coords)  
        self.alpha = torch.tensor(alpha).cuda()
        self.betas = torch.tensor(betas).cuda()

    def forward(self, logits, target):
        probs = logits.softmax(dim=1)
        pred_xy = probs @ self.coords
        true_xy = self.coords[target]

        loss_xent = F.cross_entropy(logits, target)
        loss_ord = F.mse_loss(self.betas*pred_xy, self.betas*true_xy)
        return (1-self.alpha)*loss_xent + self.alpha*loss_ord
# %%
alpha = 0.9
betas = [1,1.5]
correct = 19

loss = OrdinalLoss(class2coords, alpha, betas).cuda()

x = torch.zeros([12,39]).cuda()
x[:,correct] = 10

los_l = lambda y: loss(x, torch.Tensor([y]*12).type(torch.LongTensor).cuda()).item()

losses = [los_l(y) for y in range(39)]

losses.insert(37,np.nan)
losses.insert(34,np.nan)
losses.insert(29,np.nan)

plt.figure(figsize=(10,10))
sns.heatmap(np.array(losses).reshape(-1,6),
            square=True,xticklabels=label_f.strs[1],
            yticklabels=label_f.strs[0])
plt.yticks(rotation=0)
plt.ylabel('Temperature (Most to least)')
plt.xlabel('Luminosity (Most to least)')
plt.title(f'''Loss Values (Correct = {label_f.to_str(correct, True)})
Alpha = {alpha} | Betas = {betas}''');

# %%
# from torchmetrics.regression import MeanAbsoluteError
from models.conv1d import StarClassifier4

filters = [1,32,128,512,2048,512,128,32]
strides = [1, 2, 2, 2, 2, 2, 2, 2]

model_config = {'filter_sizes':filters,
                'kernels':[3,5,7],
                'strides':strides,
                'output_dim':len(label_f.idx_dict),
                'hidden_dim':2048,
                'dropout':0,
                'input_shape':flux_numpy.shape[-1]}

torch.manual_seed(1337)
torch.cuda.manual_seed(1337)
torch.cuda.manual_seed_all(1337)

classifier = StarClassifier4(**model_config).to('cuda')

custom_loss = OrdinalLoss(class2coords, 0.5, betas=[1,1.5]).cuda()

adamw = torch.optim.AdamW(classifier.parameters(), lr=0.00005, betas=(0.9,0.95))

# %%
folder_name = 'final9_half_ordinal'
path = f'models/weights/conv1d/cls/{folder_name}'
if not os.path.exists(path):
    os.mkdir(path)

epochs = 200

losses = []
metrics = []

best_f1 = 0.4

for i in range(epochs):
    tr_l, tr_f1, tr_qwk = trainer(train_loader, classifier, custom_loss, adamw)
    vl_l, vl_f1, vl_qwk = validator(val_loader, classifier, custom_loss)

    losses.extend([tr_l, vl_l])
    metrics.append((
        np.mean(tr_l), tr_f1, tr_qwk,
        np.mean(vl_l), vl_f1, vl_qwk
    ))

    if vl_f1 > best_f1 * 1.01:
        early_stop_counter = 0
        best_f1 = vl_f1
        print(f"New best model found at epoch {i}")
        torch.save(classifier.state_dict(), f'{path}/best_conv1d_cls_{i}.pth.tar')

    # if early_stop_counter >= 10:
        # print(f'Early stopping at epoch {i}')
        # break
    if i%5 == 0:
        printout = '\t|\t'.join([str(np.round(x,2)) for x in metrics[-1]])
        print(f'''{i+1}/{epochs} \t|\t: {printout}''')

# %%
import pickle
with open("test", "wb") as fp:   #Pickling
    pickle.dump(metrics, fp)
# %%

fig, (l_ax, f_ax) = plt.subplots(1,2, figsize=(11,5), dpi=300)
fig.suptitle('Training Graphs',fontsize=13)

l_ax.plot([x[0] for x in metrics])
l_ax.plot([x[3] for x in metrics], linestyle='--')
l_ax.legend(['Training Loss','Validation Loss'])
l_ax.set_xlabel('Epochs')
l_ax.set_ylabel('Loss')
# l_ax.set_title('Training loss');

f_ax.plot([x[1] for x in metrics], c='green')
f_ax.plot([x[4] for x in metrics],c='red',linestyle='--')
f_ax.legend(['Training F1','Validation F1'])
f_ax.set_xlabel('Epochs')
f_ax.set_ylabel('F1 Scores')
# f_ax.set_title('Training and validation F1 scores');

# %%

temp = np.mean(np.unique(
    label_f.to_regr(label_f.unique,idx=False)[:,0],
    return_counts=True)[1])

lum = np.mean(np.unique(
    label_f.to_regr(label_f.unique,idx=False)[:,1],
    return_counts=True)[1])

t_qwk_t, l_qwk_t = zip(*[x[2] for x in metrics])
t_qwk_v, l_qwk_v = zip(*[x[5] for x in metrics])

fig, ax = plt.subplots(1,1, figsize=(10,7), dpi=300)
ax.plot(np.array(t_qwk_t))
ax.plot(np.array(l_qwk_t))
ax.plot(np.array(t_qwk_v), linestyle='--')
ax.plot(np.array(l_qwk_v), linestyle='--')
ax.legend([
    'Temp QWK - Train',
    'Lum QWK - Train',
    'Temp QWK - Val',
    'Lum QWK - Val'])
ax.set_xlabel('Epochs')
ax.set_ylabel('QWK Scores')
ax.set_title('Quadratic Weighted Kappa Scores')

# %%
folder_name = 'final9_half_ordinal'
path = f'models/weights/conv1d/cls/{folder_name}'
best_i = 66

classifier3 = StarClassifier4(**model_config).to('cuda')
classifier3.load_state_dict(torch.load(f'{path}/best_conv1d_cls_{best_i}.pth.tar'))

# %%

def test_time_validator(dataloader, model, n):
    scores = []
    model.eval()
    with torch.no_grad():
        for fluxes, labels in dataloader:
            labels = labels.type(torch.LongTensor).cuda()
            fluxes = torch.roll(fluxes, n).cuda()

            class_scores = model(fluxes)
            scores.append(class_scores)

        return scores

from tqdm import auto

final_scores = []
for shift in auto.tqdm(range(-5,5,1)):
    x = test_time_validator(test_loader, classifier3, shift)
    final_scores.append(x)
# %%

norm_weights = 1/(1 * np.sqrt(2 * np.pi)) * np.exp( - (np.linspace(-3,3,10)**2) ) / (2 * 1**2)
norm_weights = (norm_weights-min(norm_weights))/(max(norm_weights)-min(norm_weights))
norm_weights = np.tile(norm_weights,(39,1)).T

final = torch.stack([torch.cat(x) for x in final_scores],dim=1)
ah = final*torch.tensor(norm_weights).cuda()
ah = ah.sum(dim=1).argmax(dim=-1)
# %%
preds, actuals = validator(test_loader, classifier3, custom_loss, test=True)

# pred_aug = list(ah.cpu().numpy())

# %%

ah2 = final.argmax(dim=-1).mode(dim=-1).values
pred_aug2 = list(ah2.cpu().numpy())
# %%
sm = stellar_metrics(label_f, preds, actuals, True, False)

sm.draw_cm('Conv1d', 300, False)

print(f'F1 Macro: {sm.f1_macro():.2%}')
print(f'Mean Absolute Error: {sm.mae():.2f}')
print(f'Q-Weighted Kappa: {sm.two_stage_qwk("q")[0]:.2%}, {sm.two_stage_qwk("q")[1]:.2%}')

# %%
def f1(set):
    test_iter = split_dataset(flux, labels, CustomTensorDataset, set)
    test_loader = DataLoader(test_iter, 256)
    preds, actuals = validator(test_loader, classifier3, custom_loss, test=True)
    sm = stellar_metrics(label_f, preds, actuals, True, False)
    f1_sc = sm.f1_macro()
    return f1_sc

from scipy.stats import bootstrap

x = bootstrap((test,),f1,n_resamples=10, random_state=10)

# %%

def qwk(set):
    test_iter = split_dataset(flux, labels, CustomTensorDataset, set)
    test_loader = DataLoader(test_iter, 256)
    preds, actuals = validator(test_loader, classifier3, custom_loss, test=True)
    sm = stellar_metrics(label_f, preds, actuals, True, False)
    l_qwk = sm.two_stage_qwk('q')[-1]
    return l_qwk

from scipy.stats import bootstrap

x = bootstrap((test,),qwk,n_resamples=10, random_state=10)

# %%

sm.report('', False)
sm.report('temp', False)
sm.report('lum', False)

# %%
sm.draw_cm('Conv1d', 300, True)

# %%

sm.draw_ord_cm('Conv1d', 300)

# %%
from models.conv1d import StarClassifier2_old
fs = [128//(2**i) for i in range(1,5)]
fs.insert(0,1)

model_config = {'filter_sizes':fs,
                'output_dim':40,
                'hidden_dim':2048,
                'dropout':0.1}

classifier2 = StarClassifier2_old(**model_config).to('cuda')
classifier2.load_state_dict(torch.load('models/weights/conv1d/cls/best_conv1d_wandb.pth.tar'))
classifier2.eval()

preds, actuals = validator(test_loader, classifier2, custom_loss, test=True)

# %%
preds[467] = 38
preds = np.array(preds)
preds[preds>14] -= 1

sm = stellar_metrics(label_f, preds, actuals, True, 'quadratic', False)

sm.draw_cm('Conv1d', 300, False)

# %%
# sm.draw_cm('Conv1d', 300, True)

# %%

# print(sm.two_stage_qwk())
print(sm.f1_macro())
print(sm.mae())
# %%
from sklearn.metrics import (
    mean_absolute_error, mean_absolute_percentage_error, f1_score,
    cohen_kappa_score, confusion_matrix, classification_report
)

f1_score(actuals, preds, average='macro')
# %%
preds2, actuals2 = validator(val_loader, classifier2, custom_loss, test=True)

f1_score(actuals2, preds2, average='macro')
# %%
hdul = fits.open('data/dataset3_subset2.fits')
dataset = hdul[1].data
hdul.close()

hdul = fits.open('data/dataset_highres_subset.fits')
dataset = hdul[1].data
hdul.close()
