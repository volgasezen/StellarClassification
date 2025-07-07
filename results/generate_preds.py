# %%
import numpy as np
import pandas as pd
from astropy.io import fits
from eval_utils import label_field, stellar_metrics
from scipy.stats import norm

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

hdul = fits.open('data/dataset3_subset2.fits')
dataset = hdul[1].data
hdul.close()

test_ind = np.load('data/3_2_test.npy')

label_f = label_field(dataset, regr=False, new=True)

classes = label_f.classes
indices = label_f.ord_to_idx(classes)

y_true = indices[test_ind]

flux_numpy = dataset.FLUX.newbyteorder().byteswap()
flux = torch.tensor(flux_numpy[test_ind]).float()
label = torch.tensor(y_true)

test_iter = TensorDataset(flux, label)
test_loader = DataLoader(test_iter, 16)

ultimate_dict = {}
# %%
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

folder_name = 'final7_starclassifier4_test'
path = f'models/weights/conv1d/cls/{folder_name}'
best_i = 148

classifier3 = StarClassifier4(**model_config).to('cuda')
classifier3.load_state_dict(torch.load(f'{path}/best_conv1d_cls_{best_i}.pth.tar'))

print('Number of parameters in conv1d best:')
num_param = sum(x.numel() for x in classifier3.parameters())
print(f"{num_param:,d}")

def generate_predictions(dataloader, model):
    preds = []
    scores = []
    model.eval()
    with torch.no_grad():
        for fluxes, labels in dataloader:
            fluxes = fluxes.cuda()

            class_scores = model(fluxes)
            class_preds = class_scores.argmax(dim=1)
            preds.extend(class_preds.tolist())
            scores.extend(class_scores.cpu().tolist())

        return {'preds':np.array(preds), 'scores':np.array(scores)}

out_dict = generate_predictions(test_loader, classifier3)

ultimate_dict.update({'conv1d_best': out_dict})

# %%
folder_name = 'final9_half_ordinal'
path = f'models/weights/conv1d/cls/{folder_name}'
best_i = 66

classifier3 = StarClassifier4(**model_config).to('cuda')
classifier3.load_state_dict(torch.load(f'{path}/best_conv1d_cls_{best_i}.pth.tar'))

print('Number of parameters in conv1d half ord:')
num_param = sum(x.numel() for x in classifier3.parameters())
print(f"{num_param:,d}")

out_dict = generate_predictions(test_loader, classifier3)

ultimate_dict.update({'conv1d_half_ord': out_dict})

# %%
folder_name = 'final10_no_ordinal'
path = f'models/weights/conv1d/cls/{folder_name}'
best_i = 65

classifier3 = StarClassifier4(**model_config).to('cuda')
classifier3.load_state_dict(torch.load(f'{path}/best_conv1d_cls_{best_i}.pth.tar'))

out_dict = generate_predictions(test_loader, classifier3)

ultimate_dict.update({'conv1d_no_ord': out_dict})

# %%
from models.conv1d import ResNet1D50

folder_name = 'resnet_test2'
path = f'models/weights/conv1d/cls/{folder_name}'
best_i = 172

classifier3 = ResNet1D50(
    num_classes=len(label_f.unique),
    in_channels=1
    ).to('cuda')
    
classifier3.load_state_dict(torch.load(f'{path}/best_conv1d_cls_{best_i}.pth.tar'))

print('Number of parameters in resnet50_1d:')
num_param = sum(x.numel() for x in classifier3.parameters())
print(f"{num_param:,d}")

out_dict = generate_predictions(test_loader, classifier3)

ultimate_dict.update({'resnet50_1d': out_dict})


# %%
from models.conv1d import StarClassifier2

fs = [1024//(2**i) for i in range(1,5)]
fs.insert(0,1)

model_config = {'filter_sizes':fs,
                'output_dim':len(label_f.idx_dict),
                'hidden_dim':2048,
                'dropout':0,
                'final_set':False}

path = 'models/weights/conv1d/cls/final6_test'
best_i = 44

classifier3 = StarClassifier2(**model_config).to('cuda')
classifier3.load_state_dict(torch.load(f'{path}/best_conv1d_cls_{best_i}.pth.tar'))

print('Number of parameters in conv1d_old:')
num_param = sum(x.numel() for x in classifier3.parameters())
print(f"{num_param:,d}")

out_dict = generate_predictions(test_loader, classifier3)

ultimate_dict.update({'conv1d_old': out_dict})

# %%
from kan_convs import KANConv1DLayer

class SimpleConvKAN(nn.Module):
    def __init__(
            self,
            layer_sizes,
            num_classes: int = 10,
            input_channels: int = 1,
            spline_order: int = 3,
            groups: int = 1):
        super(SimpleConvKAN, self).__init__()

        self.layers = nn.Sequential(
            KANConv1DLayer(input_channels, layer_sizes[0], kernel_size=3, spline_order=spline_order, groups=1, padding=1, stride=1,
                           dilation=1),
            KANConv1DLayer(layer_sizes[0], layer_sizes[1], kernel_size=3, spline_order=spline_order, groups=groups, padding=1,
                           stride=2, dilation=1),
            KANConv1DLayer(layer_sizes[1], layer_sizes[2] , kernel_size=3, spline_order=spline_order, groups=groups, padding=1,
                           stride=2, dilation=1),
            KANConv1DLayer(layer_sizes[2], layer_sizes[3], kernel_size=3, spline_order=spline_order, groups=groups, padding=1,
                           stride=1, dilation=1),
            nn.AdaptiveAvgPool1d(1)
        )
        self.output = nn.Linear(layer_sizes[3], num_classes)

        self.drop = nn.Dropout(p=0.25)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.layers(x)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        x = self.output(x)
        return x

fs = [2**(i+5) for i in range(1,5)]

torch.manual_seed(1337)
torch.cuda.manual_seed(1337)
torch.cuda.manual_seed_all(1337)

model_config = {
    'layer_sizes': fs,
    'num_classes': len(label_f.unique)
}

model = SimpleConvKAN(**model_config).cuda()

folder_name = 'kan_conv_test'
path = f'models/weights/conv1d/cls/{folder_name}'
best_i = 140

classifier3 = SimpleConvKAN(**model_config).to('cuda')
classifier3.load_state_dict(torch.load(f'{path}/best_conv_kan{best_i}.pth.tar'))

print('Number of parameters in kan_conv:')
num_param = sum(x.numel() for x in classifier3.parameters())
print(f"{num_param:,d}")

out_dict = generate_predictions(test_loader, classifier3)

ultimate_dict.update({'convkan': out_dict})
# %%

# %%
from convtran.model import ConvTran

model_config = {
    'Data_shape': (0,1,flux_numpy.shape[1]),
    'emb_size': 64,
    'dim_ff': 256,
    'num_heads': 16,
    'Fix_pos_encode': 'tAPE',
    'Rel_pos_encode': None,
    'dropout': 1e-2,
    'num_classes': len(label_f.unique)
}

folder_name = 'convtran_test3'
path = f'models/weights/convtran/{folder_name}'
best_i = 92

classifier3 = ConvTran(model_config).to('cuda')
classifier3.load_state_dict(torch.load(f'{path}/best_convtran{best_i}.pth.tar'))

print('Number of parameters in conv_tran:')
num_param = sum(x.numel() for x in classifier3.parameters())
print(f"{num_param:,d}")

def generate_predictions(dataloader, model):
    preds = []
    scores = []
    model.eval()
    with torch.no_grad():
        for fluxes, labels in dataloader:
            fluxes = fluxes.cuda()

            class_scores = model(fluxes.unsqueeze(1))
            class_preds = class_scores.argmax(dim=1)
            preds.extend(class_preds.tolist())
            scores.extend(class_scores.cpu().tolist())

        return {'preds':np.array(preds), 'scores':np.array(scores)}

out_dict = generate_predictions(test_loader, classifier3)

ultimate_dict.update({'convtran': out_dict})
# %%
#import pickle 
#
#with open('ultimate_dict.pkl', 'wb') as f:
#    pickle.dump(ultimate_dict, f)
# %%
