from __future__ import print_function


import os

import sys

import cv2
import numpy as np


sys.path.append(os.getcwd())


def resize_im(im, scale, max_scale=None):
    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])
    return cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR), f


def draw_boxes(img, image_name, boxes, scale,index):
    base_name = image_name.split('\\')[-1]
    i=0

    size = os.path.getsize(r'.\crnn\data\predict_label.txt')
    f2 = open(r'.\crnn\data\predict_label.txt', 'r+')




    f2.read()
    with open('data/results/' + 'res_{}.txt'.format(base_name.split('.')[0]), 'w') as f:

        max1 = 0
        startX1 = 0
        StartY1 = 0
        endX1 = 0
        endY1 = 0

        for box in boxes:
            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                continue
            if box[8] >= 0.9:
                color = (0, 255, 0)
            elif box[8] >= 0.8:
                color = (255, 0, 0)
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[4]), int(box[5])), color, 2)
            cv2.line(img, (int(box[6]), int(box[7])), (int(box[2]), int(box[3])), color, 2)
            cv2.line(img, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), color, 2)

            if max1 < int(box[6]) - int(box[0]):
                max1 = int(box[6]) - int(box[0])
                startX1 = int(box[0])
                StartY1 = int(box[1])
                endX1 = int(box[6])
                endY1 = int(box[7])


            min_x = min(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
            min_y = min(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))
            max_x = max(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
            max_y = max(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))

            line = ','.join([str(min_x), str(min_y), str(max_x), str(max_y)]) + '\r\n'
            f.write(line)

        region = img[StartY1:endY1, startX1:endX1, :]
        cv2.imwrite(r'./cardNum' + '\\' + str(index) + '_' + str(i) + '.png', region)



        f2.write( str(index) + '_' + str(i) + '.png'+ '\t'+'a' + '\n')

    f2.close()
    #img = cv2.resize(img, None, None, fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_LINEAR)
    #cv2.imwrite(os.path.join("./ctpn/data/results", base_name), img)
