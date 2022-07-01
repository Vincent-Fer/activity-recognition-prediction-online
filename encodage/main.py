from scipy.io import loadmat
import numpy as np
import pandas
from os.path import join,exists
from os import mkdir
import cv2
import math
import os

actions = ['No-Action', 'sweeping', 'gargling', 'opening cupboard', 'washing hands', 'eating', 'writing', 'wiping',
           'drinking','opening microwave oven', 'Throwing trash']


def get_data(file,type='foot_to_foot'):
    try:
        f = open(file,'r').read().split()
        datait = [float(x) for x in f]
        if type=='no_order':
            data = np.asarray(datait)
            data = data.reshape((25,3))
        else:
            spine_base = datait[0:3]
            spine_mid = datait[3:6]
            neck = datait[6:9]
            head = datait[9:12]
            shoulder_left = datait[12:15]
            elbow_left = datait[15:18]
            wrist_left = datait[18:21]
            hand_left = datait[21:24]
            shoulder_right = datait[24:27]
            elbow_right = datait[27:30]
            wrist_right = datait[30:33]
            hand_right = datait[33:36]
            hip_left = datait[36:39]
            knee_left = datait[39:42]
            ankle_left = datait[42:45]
            foot_left = datait[45:48]
            hip_right = datait[48:51]
            knee_right = datait[51:54]
            ankle_right = datait[54:57]
            foot_right = datait[57:60]
            spine_shoulder = datait[60:63]
            handtip_left = datait[63:66]
            thumb_left = datait[66:69]
            handtip_right = datait[69:72]
            thumb_right = datait[72:75]

            if type=='human':
                data=np.stack((head, neck, spine_shoulder, shoulder_left, shoulder_right, elbow_left, elbow_right,
                                                               wrist_left, wrist_right, thumb_left, thumb_right, hand_left, hand_right, handtip_left,
                                                               handtip_right, spine_mid, spine_base, hip_left, hip_right, knee_left, knee_right,
                                                               ankle_left, ankle_right, foot_left, foot_right))
            else :
                data=np.stack((foot_left, ankle_left, knee_left, hip_left, spine_base, handtip_left, thumb_left,
                                        hand_left, wrist_left, elbow_left, shoulder_left
                                                                   ,spine_shoulder,head,neck, shoulder_right,elbow_right,
                                                                        wrist_right,   hand_right,thumb_right
                                                                       , handtip_right, spine_mid, hip_right,
                                                                       knee_right, ankle_right,foot_right))
        return data
    except:
        print('Ex',file)
        return None

def get_labels(file):
    labels = open(file,'r').read().splitlines()
    prev_action=None
    start =[]
    end = []
    actions=[]
    for line in labels:
        if line.replace(' ','').isalpha():
            prev_action = line.strip()
        else:
            tab = line.split(' ')
            start.append(int(tab[0]))
            end.append(int(tab[1]))
            actions.append(prev_action)
    return (start,end,actions)

def get_image_label(start,end,labels):
    index = (start+end)//2
    for s,e,a in set(zip(labels[0],labels[1],labels[2])):
        if s <= index and index <= e:
            return a
    return 'No-Action'

def to_ludl(data_path,labels,window_length=30,type='no_order'):
    start_frame = min(labels[0]) - window_length//2
    end_frame = max(labels[1]) + window_length //2
    data = []
    for i in range(start_frame,end_frame+1):
        data.append(get_data(data_path+'/'+str(i)+'.txt',type))
    images = [data[i:i + window_length] for i in range(len(data) - window_length + 1)]
    lab = [get_image_label(i,i+window_length,labels) for i in range(start_frame,end_frame -window_length+2)]

    i=0
    while i <len(lab):
        if lab[i] is None:
            del lab[i]
            del images[i]
        else:
            i+=1
    i = 0
    while i < len(images):
        for x in images[i]:
            if x is None or not x.shape==(25,3):
                del lab[i]
                del images[i]
                break
        else:
            i += 1
    return np.asarray(images),lab


def transform_image_ludl(image,path,name):
    RGB = image
    height = image.shape[1]
    width = image.shape[0]
    X = np.arange(height)
    Y = np.arange(width)
    RGB = np.squeeze(RGB)
    for i in range(3):
        RGB[:,:,i] = np.floor(255 * (RGB[:,:,i] - np.amin(RGB[:,:,i])) / (np.amax(RGB[:,:,i]) - np.amin(RGB[:,:,i])))
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for i in X:
        for j in Y:
            img[i,j]=RGB[j,i]
    # img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(join(path,name+'_.png'),img)
    return img

def transform_ludl(data_path,label_path,out_path):
    images, labels = to_ludl(data_path, get_labels(label_path), window_length=30,type='foot')
    for i in range(len(images)):
        path = join(out_path,labels[i])
        if not exists(path):
            mkdir(path)
        transform_image_ludl(images[i],path,str(i))


def to_nassim(data_path,labels,window_length=40,type_='foot_to_foot'):

    # start_frame = min(labels[0]) - window_length//2
    # end_frame = max(labels[1]) + window_length //2

    subdirs = [x[2] for x in os.walk(data_path)][0]
    frames = [int(x[:-4]) for x in subdirs]
    start_frame = min(frames)
    end_frame = max(frames)
    data = []
    for i in range(start_frame,end_frame+1):
        data.append(get_data(data_path+'/'+str(i)+'.txt',type_))
    images = [data[i:i + window_length] for i in range(len(data) - window_length + 1)]
    lab = [get_image_label(i,i+window_length,labels) for i in range(start_frame,end_frame -window_length+2)]
    i=0
    while i <len(lab):
        if lab[i] is None:
            del lab[i]
            del images[i]
        else:
            i+=1
    i = 0
    while i < len(images):
        jump = False
        for x in images[i]:
            if x is None or not x.shape==(25,3):
                    del lab[i]
                    del images[i]
                    jump = True
                    break
        if not jump:
            i += 1
    return np.asarray(images),np.asarray(lab)



def transform_nassim(data_path,label_path,out_path):
    images, labels = to_nassim(data_path, get_labels(label_path), window_length=40,type_='foot')
    data = []
    lab = []
    for i in range(len(images)):
        path = join(out_path,labels[i])
        if not exists(path):
            mkdir(path)
        data.append(transform_image_ludl(images[i],path,str(i)))
        lab.append(actions.index(labels[i]))

    data = np.asarray(data)
    labels = np.asarray(lab)
    return data , labels




data_path='data'
train_path = 'Train_OAD_40'
test_path = 'Test_OAD_40'
if not exists(train_path):
    mkdir(train_path)
if not exists(test_path):
    mkdir(test_path)
train_sub = [1, 2, 3, 4, 7, 8, 9, 14, 15, 16, 18, 19, 20, 22, 23, 24, 25, 32, 33, 34, 35, 37, 38, 39, 49, 50, 51, 54, 57, 58]
test_sub  = [0, 10, 13, 17, 21, 26, 27, 28, 29, 36, 40, 41, 42, 43, 44, 45, 52, 53, 55, 56]
train = None
train_label = None
test = None
test_label = None
for i in range(59):
    path = join(data_path, str(i))
    label_path = join(path,'label','label.txt')
    image_path = join(path,'skeleton')
    print('Processing sequence num ===========>',i)
    data, label = transform_nassim(image_path, label_path, train_path if i in train_sub else test_path)
    if i in train_sub:
        if train_sub.index(i)==0:
            train = data
            train_label = label
        else:
            train = np.concatenate([train, data])
            train_label = np.concatenate([train_label, label])
    elif i in test_sub:
        if test_sub.index(i)==0:
            test = data
            test_label = label
        else:
            test = np.concatenate([test,data])
            test_label = np.concatenate([test_label,label])

from keras.utils.np_utils import to_categorical
test_label = to_categorical(test_label)
train_label = to_categorical(train_label)
np.save('train_x.npy',train)
np.save('test_x.npy',test)
np.save('train_y.npy',train_label)
np.save('test_y.npy',test_label)

print(train.shape,train_label.shape,test.shape,test_label.shape)