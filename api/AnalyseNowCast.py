import os
os.environ["HDF5_USE_FILE_LOCKING"]='FALSE'
import sys
import h5py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.patches as patches
import pandas as pd

from display.display import get_cmap

if not os.path.exists('/tmp/export'):
  os.makedirs('/tmp/export')

  # Load pretrained nowcasting models
# mse_file  = '/content/drive/MyDrive/neurips-2020-sevir/models/nowcast/mse_model.h5'
# mse_model = tf.keras.models.load_model(mse_file,compile=False,custom_objects={"tf": tf})

# style_file = '/content/drive/MyDrive/neurips-2020-sevir/models/nowcast/style_model.h5'
# style_model = tf.keras.models.load_model(style_file,compile=False,custom_objects={"tf": tf})

# mse_style_file = '/content/drive/MyDrive/neurips-2020-sevir/models/nowcast/mse_and_style.h5'
# mse_style_model = tf.keras.models.load_model(mse_style_file,compile=False,custom_objects={"tf": tf})

gan_file = '/mnt/data/neurips-2020-sevir/models/nowcast/gan_generator.h5'
gan_model = tf.keras.models.load_model(gan_file,compile=False,custom_objects={"tf": tf})

# Load a part of the test dataset
from readers.nowcast_reader import read_data
x_test,y_test = read_data('/mnt/data/neurips-2020-sevir/data/interim/nowcast_testing.h5',end=50)

## 
# Functions for plotting results
##

norm = {'scale':47.54,'shift':33.44}
hmf_colors = np.array( [
    [82,82,82], 
    [252,141,89],
    [255,255,191],
    [145,191,219]
])/255

# Model that implements persistence forecast that just repeasts last frame of input
class persistence:
    def predict(self,x_test):
        return np.tile(x_test[:,:,:,-1:],[1,1,1,12])

def plot_hit_miss_fa(ax,y_true,y_pred,thres):
    mask = np.zeros_like(y_true)
    mask[np.logical_and(y_true>=thres,y_pred>=thres)]=4
    mask[np.logical_and(y_true>=thres,y_pred<thres)]=3
    mask[np.logical_and(y_true<thres,y_pred>=thres)]=2
    mask[np.logical_and(y_true<thres,y_pred<thres)]=1
    cmap=ListedColormap(hmf_colors)
    ax.imshow(mask,cmap=cmap)


def visualize_result(models,x_test,y_test,idx,ax,labels, save_path='/tmp/export/image.png'):
    fs=12
    cmap_dict = lambda s: {'cmap':get_cmap(s,encoded=True)[0],
                           'norm':get_cmap(s,encoded=True)[1],
                           'vmin':get_cmap(s,encoded=True)[2],
                           'vmax':get_cmap(s,encoded=True)[3]}
    for i in range(0,13):
        xt = x_test[idx,:,:,i]*norm['scale']+norm['shift']
        
        ax[(i-1)][0].imshow(xt,**cmap_dict('vil'))
    ax[0][0].set_title('Inputs',fontsize=fs)
    
    pers = persistence().predict(x_test[idx:idx+1])
    pers = pers*norm['scale']+norm['shift']
    x_test = x_test[idx:idx+1]
    y_test = y_test[idx:idx+1]*norm['scale']+norm['shift']
    y_preds=[]
    for i,m in enumerate(models):
        yp = m.predict(x_test)
        if isinstance(yp,(list,)):
            yp=yp[0]
        y_preds.append(yp*norm['scale']+norm['shift'])
    
    for i in range(0,12):
        ax[i][1].imshow(y_test[0,:,:,i],**cmap_dict('vil'))
    ax[0][1].set_title('Outputs',fontsize=fs)
        
    for j in range(len(ax)):
        for i in range(len(ax[j])):
            ax[j][i].xaxis.set_ticks([])
            ax[j][i].yaxis.set_ticks([])


    ax[0][0].set_ylabel('-60 Minutes', fontsize=8)
    ax[1][0].set_ylabel('-55 Minutes', fontsize=8)
    ax[2][0].set_ylabel('-50 Minutes', fontsize=8)
    ax[3][0].set_ylabel('-45 Minutes', fontsize=8)
    ax[4][0].set_ylabel('-40 Minutes', fontsize=8)
    ax[5][0].set_ylabel('-35 Minutes', fontsize=8)
    ax[6][0].set_ylabel('-30 Minutes', fontsize=8)
    ax[7][0].set_ylabel('-25 Minutes', fontsize=8)
    ax[8][0].set_ylabel('-20 Minutes', fontsize=8)
    ax[9][0].set_ylabel('-15 Minutes', fontsize=8)
    ax[10][0].set_ylabel('-10 Minutes', fontsize=8)
    ax[11][0].set_ylabel('-5 Minutes', fontsize=8)
    ax[12][0].set_ylabel('0 Minutes', fontsize=8)

    ax[0][1].set_ylabel('+5 Minutes', fontsize=8)
    ax[1][1].set_ylabel('+10 Minutes', fontsize=8)
    ax[2][1].set_ylabel('+15 Minutes', fontsize=8)
    ax[3][1].set_ylabel('+20 Minutes', fontsize=8)
    ax[4][1].set_ylabel('+25 Minutes', fontsize=8)
    ax[5][1].set_ylabel('+30 Minutes', fontsize=8)
    ax[6][1].set_ylabel('+35 Minutes', fontsize=8)
    ax[7][1].set_ylabel('+40 Minutes', fontsize=8)
    ax[8][1].set_ylabel('+45 Minutes', fontsize=8)
    ax[9][1].set_ylabel('+50 Minutes', fontsize=8)
    ax[10][1].set_ylabel('+55 Minutes', fontsize=8)
    ax[11][1].set_ylabel('+60 Minutes', fontsize=8)
    
    
    plt.subplots_adjust(hspace=0.05, wspace=0.05)

    plt.savefig(save_path)