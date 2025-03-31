import cv2 as cv
import matplotlib.pyplot as plt

img=cv.imread('../program3/soccer.jpg')

gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
plt.imshow(gray,cmap='gray'),plt.xticks([]),plt.yticks([]),plt.show()

h=cv.calcHist([gray],[0],None,[256],[0,256])
plt.plot(h, color='r', linewidth=1), plt.show()

equ=cv.equalizeHist(gray)
plt.imshow(equ,cmap='gray'),plt.xticks([]),plt.yticks([]),plt.show()

h=cv.calcHist([equ],[0],None,[256],[0,256])
plt.plot(h,color='r',linewidth=1),plt.show()