import cv2
import numpy as np
import os
import pandas as pd




####### knn code #########

def distance(v1,v2):
    return np.sqrt(((v1-v2)**2).sum())

def knn(train ,test ,k=5):
    dist = []
    m = train.shape[0]       
    
    for i in range (m) :
       
        ix=train[i,:-1]        
        iy=train[i, -1]
        
        d=distance(test,ix)
        dist.append([d,iy])
        
    dk=sorted(dist,key=lambda x: x[0])[:k]    
    
    labels = np.array(dk)[:,-1]
    
    output = np.unique(labels, return_counts = True)
    
    index = np.argmax(output[1])
   
    return output[0][index]

##############################################################################
    
#init camera

cap=cv2.VideoCapture(0)

# face detection

face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
skip = 0

 
dataset_path = r"       "         


face_data =[] 
labels =[]     
class_id=0    

names={}  

################## load the training data ################################


for fx in os.listdir(dataset_path):

    if fx.endswith('.npy'):
       
      
        names[class_id]=fx[: -4]     
        
        data_item=np.load(dataset_path+fx)
        
        face_data.append(data_item)   
        
         
        
       
        
        target = class_id*np.ones((data_item.shape[0]))

        
        
        class_id+=1   
        labels.append(target)




face_dataset = np.concatenate(face_data,axis=0)
face_labels = np.concatenate(labels,axis =0).reshape((-1,1))


trainset = np.concatenate((face_dataset,face_labels),axis=1)   

########################### TESTING ###############################


#testing
while True:
    ret,frame=cap.read()  
    if ret==False :
        continue
    faces= face_cascade.detectMultiScale(frame,1.3,5)
    for face in faces:
        x,y,w,h=face
        
        
        offset=10
        face_section =frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))
    
        # predicted label 
        out = knn(trainset,face_section.flatten() )     
    
        # display on the screen the name and rectangle around it
        pred_name = names[int(out)]
        
        
        cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2,cv2.LINE_AA)
       
        
        
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0 ),2)
    

     
    cv2.imshow('frame',frame)
    
    key = cv2.waitKey(1) & 0xff
    if key== ord('q') :
     break
 
cap.release()
cv2.destroyAllWindows()
