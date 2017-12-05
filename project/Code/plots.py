# -*- coding: utf-8 -*-
"""
Plots images with bounding boxes

Creator: Ajinkya Khamkar

inputs: images, bbox
outputs: Plots

"""

#Plots

import matplotlib.pyplot as plt 
import numpy as np
import matplotlib.patches as patches


class plots():
    def plot_gen(images ,bbox,r,batch_size=16,figsize=(10,10)):
        
        plt.figure(figsize=figsize)
        
        bbox = bbox*120
        r = r*120
        for i in range(batch_size):
             
            #Actual Bounding box
            rect = patches.Rectangle((bbox[i,0],bbox[i,1]),
                                   bbox[i,3],bbox[i,2],
                                   linewidth=1,edgecolor='r',facecolor='none')
             
             #PPredicted Bounding Box
            rect1 = patches.Rectangle((r[i,0,0],r[i,0,1]),
                                   r[i,0,3],r[i,0,2],
                                   linewidth=1,edgecolor='g',facecolor='none',
                                   label='Label')
             
            fig,ax = plt.subplots(1)
            ax.imshow(images[i])
            ax.add_patch(rect)
            ax.add_patch(rect1)

        plt.tight_layout()
        plt.show()
