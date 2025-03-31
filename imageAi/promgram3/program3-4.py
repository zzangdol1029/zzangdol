import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img=cv.imread('../program4/JohnHancocksSignature.png', cv.IMREAD_UNCHANGED)

if img.shape[-1] == 4:
    t,bin_img=cv.threshold(img[:,:,3],0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    plt.imshow(bin_img,cmap='gray'),plt.xticks([]),plt.yticks([])
    plt.show()
    b=bin_img[bin_img.shape[0]//2:bin_img.shape[0],0:bin_img.shape[0]//2+1]
    plt.imshow(b,cmap='gray'),plt.xticks([]),plt.yticks([])
    plt.show()

    se=np.uint8([0,0,1,0,0],
                [0,1,1,1,0],
                [1,1,1,1,1],
                [0,1,1,1,0],
                [0,0,1,0,0])
    b.dilation=cv.dilate(b,se,iterations=1)
    plt.imshow(b,cmap='gray'),plt.xticks([]),plt.yticks([])
    plt.show()

    b.erosion=cv.erode(b,se,iterations=1)
    plt.imshow(b,cmap='gray'),plt.xticks([]),plt.yticks([])
    plt.show()

    b_closing=cv.erode(cv.dilate(b,se,iterations=1),se,iterations=1)
    plt.imshow(b_closing,cmap='gray'),plt.xticks([]),plt.yticks([])
    plt.show()
else:
    print("이미지 파일을 찾을수 없습니다.")