import os
import numpy as np
import glob
import tensorflow as tf
import cv2

class subject:
    def __init__(self,name,dndir,sketchdir):
        self.name=name
        self.dnfs_list=sorted(glob.glob(dndir))
        #self.sketches_list=sorted(glob.glob(sketchdir))
        self.sketches_list=sketchdir



def import_data(maindir):
    """
    ls maindir:sketch,dnfs
    returns dictionory of subjects
            subjects:key=name of folder
                    subject.dnfs_list=dnfs folder path of subject
                    subject.sketches_list=sketches path of subject
                ls dnfs path:ecoded.png [0..11]
                ls sketches path:skethces[f4,s4,t4]
    """
    # maindirmain directory eg characters
    subjects={} ##dic to hold subject name and list of paths of sketches and dnfs
    sketchdir=os.path.join(maindir,'sketch')
    dndir=os.path.join(maindir,'dnfs')
    for dir in os.listdir(sketchdir):
        temp_name=dir.split('/')[-1]
        temp_subject=subject(temp_name,os.path.join(dndir,temp_name),os.path.join(sketchdir,dir))
        subjects[temp_name]=temp_subject
    return subjects

def get_sketches(sketchdir):
    """
    sketchdir:sketch folder paths
    return: list of grayscale images #should be partioned to 4 4 4
    """
    files=glob.glob(sketchdir)
    sketches=[]
    for file in files:
        skethches.append(cv2.imread(file,'0'))
    return sketches

def get_dnfs(dnfsdir):
    """
    dnfs:dnfs folder paths
    return: list of encoded png  #should be decoded #no need to partition
    encode[d,nx,ny,nz]
    """
    files=glob.glob(dnfsdir)
    dnfs=[] #encoded png
    for file in files:
        dnfs.append(cv2.imread(file,cv2.IMREAD_UNCHANGED))
    return dnfs

def get_training_data(subjects,input_shape,output_views=12):
    sketches=np.empty(input_shape)
    views=[]list of arrays for each output view
    for 
    for x in subjects:
        dnfs.append=x.dnfs_list
