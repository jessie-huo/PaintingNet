from PIL import Image
import pandas as pd
import os
import glob
import csv
import numpy as np
def convertjpg(jpgfile,outdir,width=128,height=128):
    img=Image.open(jpgfile)
    try:
        new_img=img.resize((width,height),Image.BILINEAR)
        new_img = new_img.convert('RGBA')  
        new_img.save(os.path.join(outdir,os.path.basename(jpgfile)))
        return new_img 
    except Exception as e:
        print(e)
list = []
pri_list = []

pro_img = convertjpg("a.png","processed/")
    
list.append(np.asarray(pro_img).flatten())
    #print(len(list[0]))
#list = list.flatten()

#print(list[0][4095])
#np.savetxt("a.csv",list,delimiter = ',')

with open("test.txt", 'w', newline='') as myfile:
	for i in range(len(list)):
		for j in range(len(list[i])):
			if (j+1)%4 == 0:
				continue
			if j == (len(list[i]) - 2):
				myfile.write("%d" %list[i][j])
			else:
				myfile.write("%d," %list[i][j])
	myfile.write("\n")


with open("label.txt", 'w', newline='') as myfile:
     for i in range(len(pri_list)):
     	myfile.write("%d\n" %pri_list[i])
     
#print(len(pri_list))

