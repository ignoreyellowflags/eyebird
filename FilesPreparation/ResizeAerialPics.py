import cv2
import os
import sys
import re
import argparse
from tqdm import tqdm

parser=argparse.ArgumentParser()
parser.add_argument("-d","--directory",required=True,help="Create new directory for img 300x300")
args = vars(parser.parse_args())


def resize_300x300(folder):
    
    img_list=[]
    
    targetWidth,targetHeight=300,300
    
    root_=r'/home/eva_01/Desktop/medvedkovo/photos'
    
    os.chdir('/'.join([r'/home/eva_01/Desktop/medvedkovo/photos',folder])) 
    
    for file in os.listdir():
        res=re.findall(r'.+/.JPG',file) #find JPG files
        if res!=[]:
            img_list.append(res[0])
        else:
            pass
    #Create new directory for img 300x300
    new_dir='/'.join([os.getcwd(),str(folder)+'_300x300'])
    os.mkdir(new_dir)
    print(os.getcwd())
     
    for filename in tqdm(os.listdir()):
    	if filename.endswith('.JPG'):
	        oriimg = cv2.imread(filename,1) #reading original image
	        newimg = cv2.resize(oriimg,(targetWidth,targetHeight))#resize pic
	        cv2.imwrite('/'.join([new_dir,filename]),newimg)
        
    return 0


if __name__=='__main__':
	resize_300x300(folder=args['directory'])