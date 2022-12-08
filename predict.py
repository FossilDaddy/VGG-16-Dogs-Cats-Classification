import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image

import os
import glob
import sys


def get_files(path):
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, '*'))
    elif path.find('*') > 0:
        files = glob.glob(path)
    else:
        files = [path]

    files = [f for f in files if f.endswith('JPG') or f.endswith('jpeg') or f.endswith('jpg') or f.endswith('PNG') or f.endswith('png')]

    if not len(files):
        sys.exit('No images found by the given path!')

    return files


DOG_TEST_PATH = './dataset/test/dogs'
CAT_TEST_PATH = './dataset/test/cats'
OTHER_TEST_PATH = './dataset/test/other'

if __name__ == '__main__':
    files = get_files(DOG_TEST_PATH)
    cls_list = ['cats', 'dogs', 'other']

    # load the trained model
    net = load_model('./model/cat-dog-12.05.h5')

    # loop through all files and make predictions
    totalData, otherNum, dogNum, catNum = 0, 0, 0, 0
    for f in files:
        img = image.load_img(f, target_size=(256, 256))
        if img is None:
            continue
        totalData = totalData + 1
        x = image.img_to_array(img)
        x = preprocess_input(x)
        x = np.expand_dims(x, axis=0)
        pred = net.predict(x)[0]
        top_inds = pred.argsort()[::-1][:5]
        if cls_list[top_inds[0]] == 'other':
            otherNum = otherNum + 1
        elif cls_list[top_inds[0]] == 'dogs':
            dogNum = dogNum + 1
            # print(f)
        elif cls_list[top_inds[0]] == 'cats':
            catNum = catNum + 1
            # print(f)
        for i in top_inds:
            print('    {:.3f}  {}'.format(pred[i], cls_list[i]))
    print('Total data: {}\nother num: {}\ndogs num: {}\ncats num: {}\n'
          .format(totalData, otherNum, dogNum, catNum))
