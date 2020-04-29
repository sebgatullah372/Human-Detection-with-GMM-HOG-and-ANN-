from sklearn.svm import LinearSVC
#from sklearn.externals import joblib
import glob
import os
from skimage.feature import hog
import numpy as np
import cv2
from sklearn.externals import joblib
import pandas as pd
import random
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
#from keras.layers import Dense, Input, Dropout
from keras.callbacks import TensorBoard,EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.metrics import precision_score , recall_score , classification_report, confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt
import time

def shuffle_in_place(array):
    array_len = len(array)
    assert array_len > 2, 'Array is too short to shuffle!'
    for index in range(array_len):
        swap = random.randrange(array_len - 1)
        swap += swap >= index
        array[index], array[swap] = array[swap], array[index]
def shuffle(array):
    copy = list(array)
    shuffle_in_place(copy)
    return copy



MODEL_PATH = 'models'
RANDOM_STATE= 31

#pos_im_path = 'Hum\\pos'
#neg_im_path = 'Hum\\neg'
pos_feat= 'features\\pos'
neg_feat= 'features\\neg'
labels = []    
samp=[]
# Get positive samples
for feat_path in glob.glob(os.path.join(pos_feat, '*.feat')):
    #image = cv2.imread(filename, 0)
    #hist =  hog(image, orientations=9, pixels_per_cell=(6, 6), cells_per_block=(2, 2), block_norm='L2', visualise=False, transform_sqrt=False, feature_vector=True)
    x = joblib.load(feat_path)
    labels.append(1)
    samp.append(np.array(x[0:6480]))

# Get negative samples
for feat_path in glob.glob(os.path.join(neg_feat, '*.feat')):
    #img = cv2.imread(filename, 0)
    #hist =  hog(img, orientations=9, pixels_per_cell=(6, 6), cells_per_block=(2, 2), block_norm='L2', visualise=False, transform_sqrt=False, feature_vector=True)
    x = joblib.load(feat_path)
    #print(x)
    labels.append(0)
    samp.append(np.array(x[0:6480]))

# Shuffle Samples
#samp = shuffle(samp)
#labels= shuffle(labels)
#c= list(zip(samp,labels))
#random.shuffle(c)
#samp,labels= zip(*c)
sample= list(samp)
#label= list(labels) 

sample_df = pd.DataFrame()
sample_df["Sample"]=pd.Series(sample)
sample_df["Label"] = pd.Series(labels)

sample_df.sample(frac=1)


sample_data = sample_df["Sample"].values
sample_label = sample_df["Label"].values

#samp= samp.reshape(1,-1)
#labels= labels.reshape(1,-1)
sample_data_reshaped=[]
print(len(sample_data))
for i in sample_data:
    np.array(i,dtype='float32')
    i.reshape(1,6480)
    sample_data_reshaped.append(i)
    #print(i)
sample_data_reshaped = np.array(sample_data_reshaped)
sample_data_reshaped.reshape(len(sample_data_reshaped),6480,1)
X_train, X_test, y_train, y_test = train_test_split(sample_data_reshaped, sample_label, test_size=0.25, random_state=42)    
#x_train = sample[0:6480]
#Model
classifier = Sequential()
#First Hidden Layer
classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal', input_dim=6480))
#Second  Hidden Layer
#classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal'))

#Output Layer
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))

model_name='new.model'
bst_model_path = "F:/INRIAPerson"


classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])

classifier.fit(X_train,y_train,validation_data=(X_test,y_test),batch_size=4, epochs=5,shuffle=True)
_, eval_model=classifier.evaluate(X_train, y_train)
#predictions = classifier.predict_classes(sample_data_reshaped)
start = time.time()
predictions = classifier.predict_classes(X_test)
elapsed = (time.time()-start)
print("time",elapsed)
print(classification_report(y_test,predictions))
print("Model Accuracy",eval_model*100)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy",accuracy*100)
tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
print(tn, fp, fn, tp)
fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristics')
plt.legend(loc="lower right")
plt.show()
classifier.save(model_name)



