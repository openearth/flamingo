import os
import sys
import cv2
import re
import numpy as np
import readline

CATEGORIES = []

fpath = '/Users/hoonhout/Checkouts/PhD/Measurements/Argus/Kijkduin/Max'
imgfile1 = '1373189403.Sun.Jul.07_09_30_03.UTC.2013.kijkduin.c3.snap.jpg'
imgfile2 = '1373189403.Sun.Jul.07_09_30_03.UTC.2013.kijkduin.c3.snap.jpg'

def _normalize(arr):
    arr[np.isnan(arr)] = 0
    return np.round(
        (arr.astype(np.float32) - np.min(arr)) / (np.max(arr) - np.min(arr)) * 255).astype(np.uint8)

def complete(text, state):
    for cat in CATEGORIES:
        if cat.startswith(text):
            if not state:
                return cat
            else:
                state -= 1

if len(sys.argv) < 2:
    raise Exception('Insufficient number of paramaters, usage: classification.py <file_in>')

fin = sys.argv[1]

if not fin.endswith('.npy'):
    fin = '%s.npy' % fin

if not os.path.exists(os.path.join(fpath,fin)):
    raise Exception('File not found: %s' % fin)

if 'libedit' in readline.__doc__:
    readline.parse_and_bind("bind ^I rl_complete")
else:
    readline.parse_and_bind("tab: complete")
readline.set_completer(complete)

#time_var = _normalize(np.load(os.path.join(fpath, 'time_var.npy')))
#var_grad = _normalize(np.minimum(np.load(os.path.join(fpath, 'var_grad.npy')),10))
segments = np.load(os.path.join(fpath, fin))

#imgrgb1 = cv2.cvtColor(time_var, cv2.COLOR_GRAY2BGR)
#imgrgb2 = cv2.cvtColor(var_grad, cv2.COLOR_GRAY2BGR)

imgrgb1 = cv2.imread(os.path.join(fpath,imgfile1))
imgrgb2 = cv2.imread(os.path.join(fpath,imgfile2))

cv2.namedWindow("win",1)

n = np.max(segments)
categories = [None] * n
last_category = 'unknown'

for i in range(200,n):
    
    img1 = imgrgb1.copy()
    img2 = imgrgb2.copy()

    for j in range(img1.shape[2]):
        ch1 = img1[:,:,j]
        ch2 = img2[:,:,j]
        if j == 2:
            ch1[segments==i] = 255
            ch2[segments==i] = 255
        else:
            ch1[segments==i] = 0
            ch2[segments==i] = 0

    cv2.imshow("win", cv2.resize(img1,(1200,800)))
    cv2.waitKey(10)
    
    categories[i] = raw_input('[%03d/%03d] Category ID: ' % (i,n))

    if len(categories[i]) == 0:
        categories[i] = last_category
    else:
        last_category = categories[i]

    if not categories[i] in CATEGORIES:
        CATEGORIES.append(last_category)

cv2.destroyWindow("win")

np.save(os.path.join(fpath,re.sub('\..+$','.cat',fin)),categories)
