from PIL import Image
from PIL import ImageStat
import colorsys
import pandas as pd
import os
import glob
import csv
import numpy as np
import cv2
import threading
def most_dominant(img):
    img = img.convert('RGB')
    img = img.resize((128,128),Image.BILINEAR)
    width, height = img.size

    r_total = 0
    g_total = 0
    b_total = 0

    count = 0
    for x in range(width):
        for y in range(height):
            r, g, b = img.getpixel((x,y))
            r_total += r
            g_total += g
            b_total += b
            count += 1
    HLS = colorsys.rgb_to_hls(r_total/count, g_total/count, b_total/count)
    #print myDominantColorRGB[i]
    hueAngle = HLS[0]*360

    # Lightness
    if (HLS[1] < 0.2):
        myDominantColor = "blacks"
    elif (HLS[1] > 0.8):
        myDominantColor = "whites"
    # saturation
    elif (HLS[2] < 0.25):
        myDominantColor = "grays"
    # hue
    elif (hueAngle < 30):
        myDominantColor = "reds"
    elif (hueAngle < 90):
        myDominantColor = "yellows"
    elif (hueAngle < 150):
        myDominantColor = "greens"
    elif (hueAngle < 210):
        myDominantColor = "cyans"
    elif (hueAngle < 270):
        myDominantColor = "blues"
    elif (hueAngle < 330):
        myDominantColor = "magentas"
    else:
        myDominantColor = "reds"
        
    return myDominantColor

def brightness(img):
   im = img.convert('L')
   stat = ImageStat.Stat(im)
   return stat.mean[0]

def unique_color(img):
    img = img.resize((128,128),Image.BILINEAR)
    img = img.convert('RGB')
    width, height = img.size

    r_total = 0
    g_total = 0
    b_total = 0
    list = []
    count = 0
    for x in range(width):
        for y in range(height):
            if img.getpixel((x,y)) not in list:
                list.append(img.getpixel((x,y)))
    return len(list)
   
def cornerP(n):
   #filename ='4.jpg'
   img=np.array(n) 
   gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
   gray = np.float32(gray)
   dst = cv2.cornerHarris(gray,2,3,0.04)
   maxd=dst.max()*0.02
   cornerlist=0
   suml=0
   #cornerlist=[i for i in np.ndenumerate(dst) if i>maxd]
   unzip_lst = zip(*dst)
   for i in unzip_lst:
     for j in i:
        suml+=1
        if j>maxd:
            cornerlist+=1
   #rows,cols,depth= img.shape
   cornerPerc=cornerlist*100/float(suml)
   cornerPerc= "%.2f" % (cornerPerc) 
   cornerPerc=float(cornerPerc)
   return cornerPerc

def edgeP(n):
   img=np.array(n) 
   Edge=0;
   edges = cv2.Canny(img,100,200) 
   for i in edges:
    for j in i:
      if (j==255):
        Edge+=1
   rows,cols,depth= img.shape
   edgePerc=Edge*100/float(rows*cols)
   edgePerc= "%.2f" % (edgePerc) 
   edgePerc=float(edgePerc)
   return edgePerc




def convertjpg(jpgfile,outdir,width=32,height=32):
    img=Image.open(jpgfile)
    try:
        new_img=img.resize((width,height),Image.BILINEAR)
        new_img = new_img.convert('RGBA')  
        new_img.save(os.path.join(outdir,os.path.basename(jpgfile)))
        return new_img 
    except Exception as e:
        print(e)

def main():
    with open('processed.csv', 'w', newline='') as csvfile:
        # 建立 CSV 檔寫入器
        writer = csv.writer(csvfile)

        writer.writerow(['id', 'dominant', 'brightness','unique_color', 'cornerP', 'edgeP'])

    threads = []
    threads.append(threading.Thread(target=pro1))
    threads.append(threading.Thread(target=pro2))
    threads.append(threading.Thread(target=pro3))
    threads.append(threading.Thread(target=pro4))
    threads.append(threading.Thread(target=pro5))
    for i in range(5):
        threads[i].start()
    for i in range(5):
        threads[i].join()






def pro1():
    count = 0
    for jpgfile in glob.glob("dataset/d2/*/*.jpg"):
        print(count)
        price = os.path.basename(jpgfile)
        id = price
        img = Image.open(jpgfile)
        dominant=most_dominant(img)
        brightness_list = brightness(img)
        unique=unique_color(img)
        corner=cornerP(img)
        edge=edgeP(img)
        count = count+1

        with open('processed.csv', 'a', newline='') as csvfile:
            # 建立 CSV 檔寫入器
            writer = csv.writer(csvfile)

            #writer.writerow(['id', 'dominant', 'brightness','unique_color', 'cornerP', 'edgeP'])
            
            writer.writerow([id, dominant, brightness_list,unique,corner,edge])




def pro2():
    count = 0
    for jpgfile in glob.glob("dataset/d3/*.jpg"):
        print(count)
        price = os.path.basename(jpgfile)
        id = price
        img = Image.open(jpgfile)
        dominant=most_dominant(img)
        brightness_list = brightness(img)
        unique=unique_color(img)
        corner=cornerP(img)
        edge=edgeP(img)
        count = count+1

        with open('processed.csv', 'a', newline='') as csvfile:
            # 建立 CSV 檔寫入器
            writer = csv.writer(csvfile)

            #writer.writerow(['id', 'dominant', 'brightness','unique_color', 'cornerP', 'edgeP'])
            
            writer.writerow([id, dominant, brightness_list,unique,corner,edge])




def pro3():
    count = 0
    for jpgfile in glob.glob("dataset/d4/*.jpg"):
        print(count)
        price = os.path.basename(jpgfile)
        id = price
        img = Image.open(jpgfile)
        dominant=most_dominant(img)
        brightness_list = brightness(img)
        unique=unique_color(img)
        corner=cornerP(img)
        edge=edgeP(img)
        count = count+1

        with open('processed.csv', 'a', newline='') as csvfile:
            # 建立 CSV 檔寫入器
            writer = csv.writer(csvfile)

            #writer.writerow(['id', 'dominant', 'brightness','unique_color', 'cornerP', 'edgeP'])
            
            writer.writerow([id, dominant, brightness_list,unique,corner,edge])

def pro4():
    count = 0
    for jpgfile in glob.glob("dataset/d5/*.jpg"):
        print(count)
        price = os.path.basename(jpgfile)
        id = price
        img = Image.open(jpgfile)
        dominant=most_dominant(img)
        brightness_list = brightness(img)
        unique=unique_color(img)
        corner=cornerP(img)
        edge=edgeP(img)
        count = count+1

        with open('processed.csv', 'a', newline='') as csvfile:
            # 建立 CSV 檔寫入器
            writer = csv.writer(csvfile)

            #writer.writerow(['id', 'dominant', 'brightness','unique_color', 'cornerP', 'edgeP'])
            
            writer.writerow([id, dominant, brightness_list,unique,corner,edge])
def pro5():
    count = 0
    for jpgfile in glob.glob("dataset/d6/*.jpg"):
        print(count)
        price = os.path.basename(jpgfile)
        id = price
        img = Image.open(jpgfile)
        dominant=most_dominant(img)
        brightness_list = brightness(img)
        unique=unique_color(img)
        corner=cornerP(img)
        edge=edgeP(img)
        count = count+1

        with open('processed.csv', 'a', newline='') as csvfile:
            # 建立 CSV 檔寫入器
            writer = csv.writer(csvfile)

            #writer.writerow(['id', 'dominant', 'brightness','unique_color', 'cornerP', 'edgeP'])
            
            writer.writerow([id, dominant, brightness_list,unique,corner,edge])


    #print(len(list[0]))
#list = list.flatten()

#print(list[0][4095])
#np.savetxt("a.csv",list,delimiter = ',')


     
#print(len(pri_list))

if __name__ == "__main__":
    main()