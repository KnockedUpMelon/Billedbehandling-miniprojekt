import cv2
import numpy as np

class HSV:
    def hsvConversion(r,g,b):
        r=r/255
        g=g/255
        b=b/255


        small=min(r,g,b)
        V=max(r,g,b)
        delta=V-small
        if (delta==0):
            H=0
        elif ((V==r)and(g>=b)):
            H=HSV.cosinusrelation(g,b,delta,0)
        elif(g==V):
            H=HSV.cosinusrelation(b,r,delta,2)
        elif(b==V):
            H=HSV.cosinusrelation(r,g,delta,4)
        elif((V==r)and(g<b)):
            H=HSV.cosinusrelation(r,b,delta,5)

        if(V==0):
            S=0
        else:
            S=(delta)/V*255
        H=H/2
        V=V*255
        return [round(H)% 180,round(S),round(V)]
    
    def cosinusrelation(val1,val2,delta,row):
        return(((val1-val2)/(delta)+row)*60)% 360

    def hsvImage(img):
        height=len(img)
        width=len(img[0])
        hsv = np.zeros((height, width, 3), dtype=np.float32)
        for y in range(height):
            for x in range(width):
                b, g, r = img[y, x]
                hsv[y, x] = HSV.hsvConversion(int(r), int(g), int(b))
        return hsv