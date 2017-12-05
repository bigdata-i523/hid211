'''
Script to Load data

Creator: Ajinkya Khamkar

HID: 211

images: holds images
annotations: holds bounding box annotations
labels: holds image labels
original: holds original bounding box locations


load_image_labels: Load images annotations and labels 

reshape_annotations: reshapes bounding box annotations

'''

import os
import numpy as np
import pandas as pd
from PIL import Image
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import gc

class load():
        
    def load_images_labels(path,count_imgs,w,h):
        
        gc.collect()
        
        images = np.ndarray (shape=(count_imgs,w,h,3),dtype = np.float32)
        annotations = pd.DataFrame()
        orig = []
        labels = []
        count  = 0
        
        for _, dirnames,_ in os.walk(path):
                
            for diri in dirnames:
                    
                for ro,_,filenames in os.walk(path+str(diri)):
                    
                    for files in filenames:
                        if files.endswith(".jpg"):
                            filepath = os.path.join(ro, files)
                            image = load_img(filepath)
                            labels.append(str(diri))
                            r,c,ch=img_to_array(image).shape
                            orig.append((r,c))
                            image=image.resize((w,h),Image.LANCZOS)
                            
                            image=img_to_array(image)
                        
                            images[count]=image
                            count+=1
                        
                            if count%250==0:
                                print (str(count)+" images loaded")
                
                df1 = pd.read_table(path+str(diri)+"/groundtruth.txt",sep=',',header =None)
                annotations = annotations.append(df1)
                
        orig = np.hstack(orig)        
        encoder = LabelEncoder()
        encoder.fit(labels)
        labels_y = encoder.transform(labels)    
        labels = to_categorical(labels_y)
                            
        return (images,labels,annotations.as_matrix(),orig)                       
    
    def reshape_annotations(annotations,orig,w,h):
        bbox = []
        width = annotations[:,2]*w/orig[:,1]
        height = annotations[:,3]*h/orig[:,0]
        xmin = annotations[:,0]*w/orig[:,1]
        ymin = annotations[:,1]*h/orig[:,0]
        bbox = np.column_stack((xmin, ymin,height,width))

        print ("annotations reshaped")
        return (bbox)
    