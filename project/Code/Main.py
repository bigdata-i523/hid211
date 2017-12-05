#Main file

'''
Creator : Ajinkya Khamkar
HID: 211

Main file for program

Pass the following arguement parsers while running

train/test variable: 1 for training from scratch, 0 for cross validation
path: Path to dataset
modelpath: path to model

Run:
    
    python3 Main.py 0 'path_to_dataset/2013/' 'path_to_model/' 

'''
import argparse
from Load import load
import numpy as np
from Model import Model_
from keras.models import load_model
from plots import plots

def Run(args):
    
    # Load images, annotations and labels
    images,labels,annotations,orig = load.load_images_labels(args.path,5681,120,120)

    #reshape original dimensions

    orig = orig.reshape([5681,2])
    
    #resize bounding boxes
    bbox = load.reshape_annotations(annotations,orig,120,120)

    images = images[:5681,:,:,:]

    #Normalize Images and bboxes
    
    bbox = bbox/120.0
    images = images/255.0
    rolledbbox = np.reshape(np.roll(bbox,-1,0),[5681,1,4])
    

    #Attach labels and annotations for training

    
    
    

    y_train = np.concatenate((bbox,labels),axis = 1)
    
    if args.train ==1:
        
        print ("Train phase")

        #Initialize Model

        model = Model_.model()
        
        #begin training
        
        Model_.fit_(images,y_train,rolledbbox,model)
        
    if args.train==0:
        
        print ("Test phase")
        
        #load model
        
        model = load_model(args.modelpath+"model1.h5")
        
        #predict bounding boxes
        
        for i in range(0,5):
            i = np.random.randint(0,images.shape[0]-128)
            x = images[i:i + 127,:,:,:]
            y = y_train [i:i + 127,:]
            
            _,_,_,_,r,_ = model.predict(x)
            plots.plot_gen(x,y,r)
            
            
    
def main():
    
    parser = argparse.ArgumentParser(description='Console:')
    
    parser.add_argument('train', metavar='train',type=int,help='Train model from scratch 1, else 0')
    
    parser.add_argument('path', metavar='path',type=str,help='Path to load Images and annotations')
    
    parser.add_argument('modelpath', metavar='modelpath',type=str,help='Path to load model for testing')
	
    args= parser.parse_args()
    
    Run(args)
    
main()


