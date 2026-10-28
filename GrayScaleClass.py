import cv2
import numpy as np



class GrayScaling:
    def greyScaleConversion(r,g,b,weights,weight):
        if (weight==True):
            gray=(weights[2]*r+weights[1]*g+weights[0]*b)
        else:
            gray=(int(r)+int(g)+int(b))/3

        return  np.uint8(round(gray))

    def grayScale(img,weights=(0.114,0.587,0.299),weightChoice=True):
        grayscale1=[]

        for x, row in enumerate(img):
            newRow = []
            for y, col in enumerate(row):
                newRow.append(GrayScaling.greyScaleConversion(col[2],col[1],col[0],weightChoice,weights))
            grayscale1.append(newRow)
        grayscale1 = np.array(grayscale1, dtype=np.uint8)
        return np.array(grayscale1, dtype=np.uint8)