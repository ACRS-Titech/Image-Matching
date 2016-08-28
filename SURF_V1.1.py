import cv2
import numpy as np
import scipy 
import matplotlib.pyplot as plt

"""please input 2 images in this function, 

   the 1st is the master image and the 2nd is the slave image."""

def surf_matching(image_1,image_2):
    
    # Load the images
    image = cv2.imread('{}'.format(image_1))

    # Convert the images to grayscale
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Surf extration
    surf = cv2.xfeatures2d.SURF_create(10000)
    (kp1, desc1) = surf.detectAndCompute(gray_img, None)

    # Setting up samples and responses for KNN
    samples = np.array(desc1)
    responses = np.arange(len(kp1), dtype = np.float32)

    # KNN training
    knn = cv2.ml.KNearest_create()
    knn.train(samples,cv2.ml.ROW_SAMPLE, responses)

    # Loading a template image and searching for similar keypoints
    template = cv2.imread('{}'.format(image_2))
    templateg = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    (kp2, desc2) = surf.detectAndCompute(templateg, None)
    
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1,desc2, k=2)
    
    matchesMask = [[0,0] for i in range(len(matches))]
    
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]
    
    draw_params = dict(matchColor = (0,255,0),
                       singlePointColor = (255,0,0),
                       matchesMask = matchesMask,
                       flags = 0)
    
    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(gray_img,kp1,template,kp2,matches,None,**draw_params)
    
    plt.imshow(img3),plt.show 
    

"""
for h, des in enumerate(desc):
des = np.array(des, np.float32).reshape((1, len(des)))
retval, results, neigh_resp, dists = knn.findNearest(des, 1)
res, dist = int(results[0][0]),dists[0][0]
    
if dist < 0.1: 
# draw matched points in red color
color = (0,0,255)
        
else:
# draw unmatched points in green color
color = (0,255,0)
    
# Draw matched key points on original image
x,y = kp[res].pt
center = (int(x),int(y))
cv2.circle(image,center,2,color,-1)
    
# Draw matched key points on template image
x,y = kp[res].pt
center = (int(x),int(y))
cv2.circle(template,center,2,color,-1)    

cv2.imshow('img',image)
cv2.imshow('template', template)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

