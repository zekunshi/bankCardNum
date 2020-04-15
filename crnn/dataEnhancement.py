import os

import matplotlib.pyplot as plt

#样本
import cv2
from PIL import ImageEnhance, Image
samplePath = r'.\crnn\data\test_images'
allSamples = os.listdir(samplePath)
i=0
maxK=4
while i<len(allSamples):
    print('samplpe '+str(i))
    j = 0
    while j<5:
        k=0
        while k<4:
            l=0
            while l<4:
                img = Image.open(samplePath + '//' + allSamples[i])
                (width, height)=img.size
                img = img.resize((int(width * (0.6 + l * 0.4)), int(height * (0.5 + l * 0.4))))

                enh_bri = ImageEnhance.Brightness(img)
                brightness = 0.8 + k * 0.2
                img1 = enh_bri.enhance(brightness)

                enh_col = ImageEnhance.Color(img1)
                color = 0.8 + j * 0.1
                img2 = enh_col.enhance(color)
                img2.save(r'.\crnn\data\images' + '\\' + allSamples[i][0:4] + '_' + str(k + j * 8) + '.png')
                print('generate ' + str(l+ 4 *k + j * 16))
                l+=1
            k+=1
        j+=1


    i+=1

