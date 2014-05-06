import os
import sys
import cv2
import re
import numpy as np
import readline
import glob

CATEGORIES = []

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

fpath = sys.argv[1]

if not os.path.exists(os.path.join(fpath)):
    raise Exception('File not found: %s' % fpath)

if 'libedit' in readline.__doc__:
    readline.parse_and_bind("bind ^I rl_complete")
else:
    readline.parse_and_bind("tab: complete")
readline.set_completer(complete)

for fname in glob.glob(os.path.join(fpath,'segments_*.npy')):

    m = re.findall('_([a-z]+)(\d+)', fname)
    s = {k:int(v) for k,v in m}

    out = os.path.join(fpath, 'categories_compactness%d_n%d_start%d_length%d' % (s['compactness'],s['n'],s['start'],s['length']))

    if not os.path.exists(out):
        print fname

        time_var = _normalize(np.load(os.path.join(fpath, 'time_var_start%d_length%s.npy' % (s['start'],s['length']))))
        var_grad = _normalize(np.load(os.path.join(fpath, 'var_grad_start%d_length%s.npy' % (s['start'],s['length']))))

        segments = np.load(os.path.join(fname))

        imgrgb1 = cv2.cvtColor(time_var, cv2.COLOR_GRAY2BGR)
        imgrgb2 = cv2.cvtColor(var_grad, cv2.COLOR_GRAY2BGR)

        cv2.namedWindow("win",1)

        n = np.max(segments)
        categories = [None] * n
        last_category = 'unknown'

        for i in range(n):
    
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

            cv2.imshow("win", np.concatenate((img1,img2),axis=1))
            cv2.waitKey(10)
    
            categories[i] = raw_input('[%03d/%03d] Category ID: ' % (i,n))

            if len(categories[i]) == 0:
                categories[i] = last_category
            else:
                last_category = categories[i]
                
            if not categories[i] in CATEGORIES:
                CATEGORIES.append(last_category)

        np.save(out,categories)

cv2.destroyWindow("win")


