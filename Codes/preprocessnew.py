import os
import cv2
import numpy as np
import pandas as pd
import mahotas as mt
from matplotlib import pyplot as plt

ds_path = "db\\train"
img_files = os.listdir(ds_path)

def create_dataset():
    names = ['area','perimeter','','equi_diameter','average_rect','','aspect_ratio','rectangularity', 'circularity','compactness','','color_mean','color_std',
    'contrast','correlation','inverse_difference_moments', 'entropy']
    df = pd.DataFrame([], columns=names)
    for file in img_files:
        imgpath = ds_path + "\\" + file
        main_img = cv2.imread(imgpath)

        #Preprocessing
        img = cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB)
        gs = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gs, (25,25),0)
        ret_otsu,im_bw_otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        kernel = np.ones((50,50),np.uint8)
        closing = cv2.morphologyEx(im_bw_otsu, cv2.MORPH_CLOSE, kernel)

        #Shape features
        contours, hierarchy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        M = cv2.moments(cnt)

        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt,True)
        equi_diameter = int(np.sqrt(4*area/np.pi))
        rect = cv2.minAreaRect(cnt)
        parta = rect[0]
        partb = rect[1]
        x1 = parta[0]
        y1 = parta[1]
        x2 = partb[0]
        y2 = partb[1]
        average = int(x1+x2+y1+y2)


        x,y,w,h = cv2.boundingRect(cnt)
        aspect_ratio = float(w)/h
        rectangularity = w*h/area
        circularity = ((perimeter)**2)/area
        compactness = (4 * (np.pi) * area)/(perimeter**2)


        #Color features
        red_channel = img[:,:,0]
        green_channel = img[:,:,1]
        blue_channel = img[:,:,2]
        blue_channel[blue_channel == 255] = 0
        green_channel[green_channel == 255] = 0
        red_channel[red_channel == 255] = 0

        red_mean = np.mean(red_channel)
        green_mean = np.mean(green_channel)
        blue_mean = np.mean(blue_channel)
        color_mean = int((red_mean + green_mean + blue_mean))

        red_std = np.std(red_channel)
        green_std = np.std(green_channel)
        blue_std = np.std(blue_channel)
        color_std = int((red_std + green_std + blue_std))

        #Texture features
        textures = mt.features.haralick(gs)
        ht_mean = textures.mean(axis=0)
        contrast = ht_mean[1]
        correlation = ht_mean[2]
        inverse_diff_moments = ht_mean[4]
        entropy = ht_mean[8]


        vector = [area,perimeter,0,equi_diameter,average,1,aspect_ratio,rectangularity,circularity,compactness,2,color_mean, color_std,contrast,
        correlation,inverse_diff_moments,entropy]

        df_temp = pd.DataFrame([vector],columns=names)
        df = df.append(df_temp)
        print(file)
    return df

dataset = create_dataset()

dataset.shape

type(dataset)

dataset.to_csv("medicinal_plant.csv")
