import numpy as np
import pandas as pd
import os
import string

dataset = pd.read_csv("medicinal_plant.csv")

dataset.head(5)

type(dataset)

maindir = r'C:\Users\VYOMA\Desktop\project\final'
ds_path = maindir + "\\db\\train"
img_files = os.listdir(ds_path)

breakpoints = [1001,1047,1048,1132,1133,1215,1216,1280,1281,1358,1359,1412]

target_list = []
for file in img_files:
    target_num = int(file.split(".")[0])
    flag = 0
    i = 0
    for i in range(0,len(breakpoints),2):
        if((target_num >= breakpoints[i]) and (target_num <= breakpoints[i+1])):
            flag = 1
            break
    if(flag==1):
        target = int((i/2))
        target_list.append(target)

y = np.array(target_list)
y

X = dataset.iloc[:]

X.head(5)


y[0:5]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 142)


X_train.head(5)

y_train[0:5]

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

X_train[0:5]

y_train[0:5]

from sklearn import svm

clf = svm.SVC(gamma='auto')
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

from sklearn import metrics

metrics.accuracy_score(y_test, y_pred)


from sklearn.model_selection import GridSearchCV

parameters = [{'kernel': ['rbf'],
               'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5],
               'C': [1, 10, 100, 1000]},
              {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}
             ]

svm_clf = GridSearchCV(svm.SVC(decision_function_shape='ovr', gamma='auto'), parameters, cv=5)
svm_clf.fit(X_train, y_train)

svm_clf.best_params_

means = svm_clf.cv_results_['mean_test_score']
stds = svm_clf.cv_results_['std_test_score']


y_pred_svm = svm_clf.predict(X_test)

metrics.accuracy_score(y_test, y_pred_svm)

import matplotlib.pyplot as plt


import os
import cv2


def bg_sub(filename):
    test_img_path = 'db\\test\\' + filename
    main_img = cv2.imread(test_img_path)
    img = cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB)

    return img

filename = input('Enter file name: ')
bg_rem_img = bg_sub(filename)



import mahotas as mt

def feature_extract(img):
    names=['','area','perimeter','','equi_diameter','average_rect','','aspect_ratio','rectangularity', 'circularity','compactness','','color_mean','color_std',
    'contrast','correlation','inverse_difference_moments', 'entropy']
    df = pd.DataFrame([], columns=names)

    #Preprocessing
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

    vector = [0,area,perimeter,0,equi_diameter,average,1,aspect_ratio,rectangularity,circularity,compactness,2,color_mean, color_std,contrast,
    correlation,inverse_diff_moments,entropy]

    df_temp = pd.DataFrame([vector],columns=names)
    df = df.append(df_temp)

    return df


features_of_img = feature_extract(bg_rem_img)


scaled_features = sc_X.transform(features_of_img)

y_pred_mobile = svm_clf.predict(scaled_features)


common_names = ['Nyctanthes Arbor - Tristis' , 'Lemon Leaf','Neem','Basil','Curry leaf','Mango Leaf']
plant = common_names[y_pred_mobile[0]]

output = plant

import webbrowser

if output == 'Nyctanthes Arbor - Tristis':
    webbrowser.open("file:///C:/Users/VYOMA/Desktop/project/final/html/Tristis.html", new=1)

elif output == 'Lemon Leaf':
    webbrowser.open("file:///C:/Users/VYOMA/Desktop/project/final/html/lemon.html", new=1)

elif output == 'Neem':
    webbrowser.open("file:///C:/Users/VYOMA/Desktop/project/final/html/neem.html", new=1)

elif output == 'Basil':
        webbrowser.open("file:///C:/Users/VYOMA/Desktop/project/final/html/basil.html", new=1)

elif output == 'Curry leaf':
        webbrowser.open("file:///C:/Users/VYOMA/Desktop/project/final/html/curry.html", new=1)

elif output == 'Mango Leaf':
        webbrowser.open("file:///C:/Users/VYOMA/Desktop/project/final/html/mango.html", new=1)

else:
    print("Unrecognized")


