import cv2 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps

X=np.load('image.npz')['arr_0']
y=pd.read_csv('labels.csv')['labels']
classes=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
nclasses=len(classes)

X_train,X_test,Y_train,Y_test=train_test_split(X,y,random_state=0,train_size=7500,test_size=2500)
X_train_scaled=X_train/255.0
X_test_scaled=X_test/255.0

classi=LogisticRegression(solver='saga',multi_class='multinomial').fit(X_train_scaled,Y_train)

y_pred=classi.predict(X_test_scaled)
accuracy=accuracy_score(Y_test,y_pred)
print(accuracy)

cap=cv2.VideoCapture(0)

while(True):
    try:
        ret,frame=cap.read()
        g=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        height,width=g.shape
        upper_left=(int(width/2-56),int(height/2-56))
        bottom_right=(int(width/2+56),int(height/2+56))
        cv2.rectangle(g,upper_left,bottom_right,(0,255,0),2)
        roi=g[upper_left[1]:bottom_right[1],upper_left[0]:bottom_right[0]]
        im_pil=Image.fromarray(roi)
        im_bw=im_pil.convert('L')
        im_bw_resized=im_bw.resize((28,28),Image.ANTIALIAS)
        im_bw_resized_inverted=PIL.ImageOps.invert(im_bw_resized)
        pixel_filter=20
        min_pixel=np.percentile(im_bw_resized_inverted,pixel_filter)
        im_bw_resized_inverted_scaled=np.clip(im_bw_resized_inverted-min_pixel,0,255)
        max_pixel=np.max(im_bw_resized_inverted)
        im_bw_resized_inverted_scaled=np.asarray(im_bw_resized_inverted_scaled)/max_pixel
        test_sample=np.array(im_bw_resized_inverted_scaled).reshape(1,784)
        test_pred=classi.predict(test_sample)
        print('predicted class=',test_pred)
        cv2.imshow('frame',g)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e :pass

cap.release()
cv2.destroyAllWindows()