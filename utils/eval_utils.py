import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm
import seaborn as sns

from sklearn.metrics import (
    mean_absolute_error, mean_absolute_percentage_error, f1_score,
    cohen_kappa_score, confusion_matrix, classification_report
)

class label_field:
    def __init__(self, dataset, regr, new):

        cols = ['SPEC', 'SUB', 'LUM']
        if new:
            cols = [f'{x}_NUM' for x in cols]
        
        lum_class = np.round(dataset[cols[2]])
        lum_class[lum_class < 1] = 1
        lum_class -= 1

        self.classes = dataset[cols[0]] * 10 + lum_class

        if regr:
            self.labels = np.vstack([
                dataset[cols[0]]+dataset[cols[1]]/10, 
                dataset[cols[2]]]).T

        else:
            self.labels = np.vstack([dataset[cols[0]], dataset[cols[2]]]).T

        self.unique = sorted(np.unique(self.classes))

        self.ord_dict = dict(
            zip(np.unique(self.classes),range(len(self.unique)))
        )
        self.idx_dict = {v: k for k, v in self.ord_dict.items()}

        self.strs = [list('OBAFGKM'), 
            'I|II|III|IV|V|VI'.split('|')]

    def ord_to_idx(self, labels):
        return np.vectorize(self.ord_dict.get)(labels)
        
    def idx_to_ord(self, indices):
        return np.vectorize(self.idx_dict.get)(indices)
    
    def __str_gen__(self, x, idx):
        main, lums = self.strs
        if idx:
            x = str(self.idx_dict.get(x))
        else:
            x = str(x)
        if x.startswith('0') | len(x) == 3:
            x = x.split('.')[::-1]
            return main[int(x[0])]+lums[int(x[1])]
        else:
            s, l = x[0], x[1]
            return main[int(s)]+lums[int(l)]
    
    def to_str(self,arr,idx):
        return np.vectorize(self.__str_gen__)(arr, idx)

    def __str_degen__(self, x, idx):
        main, lums = self.strs

        temp, lum = x[0], x[1:]
        t, l = int(main.index(temp)), int(lums.index(lum))
        
        x = t*10+l

        if idx:
            x = self.ord_to_idx(x)
        
        return x

    def from_str(self,arr,idx):
        return np.vectorize(self.__str_degen__)(arr, idx)
    
    def conv_num(self, x):
        if 3.5 <= x < 4: 
            return np.floor(x)
        elif 4 <= x < 5: 
            return np.floor(x)+1 
        else: 
            return np.round(x)

    def to_class(self,arr,str,idx):
        if ~isinstance(arr, np.ndarray):
            arr = np.array(arr)

        arr[arr > 6] = 6
        spec = np.floor(arr[:,0])
        spec[spec < 0] = 0

        sub = np.floor(arr[:,0]*10 % 10)

        lum = arr[:,1]
        m6 = spec == 6
        lum[m6] = np.vectorize(self.conv_num)(lum[m6])
        lum[~m6] = np.round(lum[~m6])
        sd_gk = (spec == 4) | (spec == 5)
        lum[sd_gk & (lum > 5)] = 5
        lum[lum < 1] = 1
        lum -= 1

        scale = np.array([10,1])
        ordi = np.sum(np.vstack([spec, lum]).T*scale,axis=1)

        if str:
            main = self.to_str(ordi, idx=False)
            w_sub = [x[:1]+str(y)+x[1:] for x,y in zip(main, sub)]
            return w_sub
        else:
            if idx:
                return self.ord_to_idx(ordi)
            else:
                return ordi

    def to_regr(self,arr,idx):
        if idx:
            arr = self.idx_to_ord(arr)
        temp, lum = np.vectorize(lambda x: (x//10,x%10))(arr)
        return np.vstack([temp,lum]).T

class stellar_metrics:
    def __init__(self, label_conv, preds, labels, idx, regr):
        self.idx = idx
        self.conv = label_conv
        self.main, self.lums = label_conv.strs

        if regr:
            self.preds_regr = preds
            self.labels_regr = labels
            self.preds = self.conv.to_class(preds, str=False, idx=idx)
            self.labels = self.conv.to_class(labels, str=False, idx=idx)
            
        else:
            self.preds_regr = self.conv.to_regr(preds, idx=idx)
            self.labels_regr = self.conv.to_regr(labels, idx=idx)
            self.preds = preds
            self.labels = labels

        if idx:
            values = self.conv.ord_to_idx(label_conv.classes)
        else:
            values = label_conv.classes

        weights = pd.Series(values).value_counts()
        self.sample_weights = 1/weights.loc[self.labels]

        self.unique = self.conv.to_str(label_conv.unique, False)

    def invert_labels(self,array,string,idx):
        if idx:
            array = self.conv.idx_to_ord(array)

        inverted = []
        if not string:
            for label in array:
                label = int(label)
                if len(str(label)) == 1:
                    inverted.append(int(f'{label}0'))
                else:
                    inverted.append(int(str(label)[::-1]))
        else:
            for label in array:
                inverted.append(label[1:]+label[0])

        return inverted

    def two_stage_qwk(self, weighting):
        if weighting == 'q':
            weighting = 'quadratic'
        elif weighting == 'l':
            weighting = 'linear'
        else:
            raise ValueError('Either q or l should be inputted.')

        temp = cohen_kappa_score(
            self.preds, 
            self.labels, 
            weights=weighting,
            sample_weight=self.sample_weights
        )

        lum = cohen_kappa_score(
            self.invert_labels(self.preds, False, self.idx), 
            self.invert_labels(self.labels, False, self.idx), 
            weights=weighting,
            sample_weight=self.sample_weights
        )

        return temp, lum

    def ordered_cm(self, invert):
        if invert:
            labs = self.invert_labels(self.unique, True)
            sort_key = lambda x: (self.lums.index(x[:-1]),self.main.index(x[-1]))
            labs = sorted(labs, key=sort_key)
            labels = self.conv.to_str(self.labels, idx=self.idx)
            preds = self.conv.to_str(self.preds, idx=self.idx)

            cm = confusion_matrix(
                self.invert_labels(labels, True), 
                self.invert_labels(preds, True), labels=labs)
        else:
            labs = self.unique
            cm = confusion_matrix(
                self.conv.to_str(self.labels, idx=self.idx), 
                self.conv.to_str(self.preds, idx=self.idx), labels=labs)
        
        return cm, labs
    
    def draw_cm(self, title, dpi, invert):
        cm, labs = self.ordered_cm(invert)

        plt.figure(dpi=dpi, figsize=(10,10))
        plt.title(title)
        ax = sns.heatmap(cm, xticklabels=labs, yticklabels=labs,
            annot=True, fmt='.4g', cmap='pink_r', square=True, 
            mask=cm==0, cbar=False, norm='log')
        
        if invert:
            sizes = pd.Series([x[1:] for x in self.unique]).value_counts().loc[self.lums].values
        else:
            sizes = pd.Series([x[0] for x in self.unique]).value_counts().loc[self.main].values

        x, y = 0, 0

        for s in sizes:
            w, h = s, s
            plt.axvline(x=x, c='k', lw=1)
            plt.axhline(y=y, c='k', lw=1)
            ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor='crimson', lw=2, clip_on=False, zorder=10))
            x += s    
            y += s
        plt.axvline(x=x, c='k', lw=1)
        plt.axhline(y=y, c='k', lw=1)
        ax.tick_params(length=0)

        plt.xlabel('Prediction')
        plt.ylabel('Ground Truth');

    def draw_ord_cm(self, title, dpi):
        diff = self.labels_regr.astype(int)-self.preds_regr.astype(int)
        cm = confusion_matrix(diff[:,1], diff[:,0])

        a, b = min(diff[:,0]), max(diff[:,0])+1
        c, d = min(diff[:,1]), max(diff[:,1])+1
        ranges = range(min(a,c), max(b,d))

        plt.figure(dpi=dpi, figsize=(10,10))
        plt.title(title)
        sns.heatmap(
            cm, annot=True, fmt='.0f', mask=cm==0, 
            xticklabels=ranges, yticklabels=ranges,
            norm=LogNorm(clip=False), cbar=False, cmap='pink_r', 
            square=True, linewidths=.1, linecolor='k'
        )
        plt.xlabel('Temperature Axis')
        plt.ylabel('Luminosity Axis');
    
    def report(self, level, out_dict):
        labs = self.conv.to_str(self.labels, idx=self.idx)
        preds = self.conv.to_str(self.preds, idx=self.idx)
        lx = self.unique
        
        if level == 'temp':
            labs = [x[0] for x in labs]
            preds = [x[0] for x in preds]
            lx = self.main
        elif level == 'lum':
            labs = [x[1:] for x in labs]
            preds = [x[1:] for x in preds]
            lx = self.lums
        
        report = classification_report(
            labs, preds, labels=lx, 
            digits=4, output_dict=out_dict, 
            zero_division=0)
            
        if out_dict:
            return report
        else:
            return print(report) 

    def mae(self, weight, sep):
        return mean_absolute_error(
            self.labels_regr, 
            self.preds_regr,
            sample_weight = self.sample_weights if weight else None,
            multioutput = 'raw_values' if sep else 'uniform_average'
            )
    
    def mape(self, weight, sep):
        labels = self.labels_regr
        labels[labels == 0] += 1e-1
        return mean_absolute_percentage_error(
            labels, 
            self.preds_regr,
            sample_weight = self.sample_weights if weight else None,
            multioutput = 'raw_values' if sep else 'uniform_average'
            )

    def f1_macro(self):
        return f1_score(self.labels, self.preds, average='macro')

    def summary(self):
        f1_macro = self.f1_macro()
        mae = self.mae(True, False)
        (k_t, k_l) = self.two_stage_qwk('q')
        return {
            'f1': f1_macro,
            'mae': mae,
            'qwk': (k_t, k_l),
        }
    