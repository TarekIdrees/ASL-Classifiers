
"""
Original file is located at
    https://colab.research.google.com/drive/1DcO-pbLwy6Flm_49LP7Iu_R_3jguFV4w
"""


"""**Set up colab for ASL data from kaggel**"""

! pip install kaggle

! mkdir ~/.kaggle

! cp kaggle.json ~/.kaggle/

! chmod 600 ~/.kaggle/kaggle.json

! kaggle datasets download grassknoted/asl-alphabet

! unzip asl-alphabet.zip

"""**Import needed libraries**"""

import numpy as np
import os
from sklearn.utils import shuffle
import cv2
import keras
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression

train_dir = 'asl_alphabet_train//asl_alphabet_train'
test_dir = 'asl_alphabet_test//asl_alphabet_test'
labels_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11,
               'M': 12,
               'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23,
               'Y': 24,
               'Z': 25, 'space': 26, 'del': 27, 'nothing': 28}

def imageProcessing(temp_imges,labels):
    images = []
    flat_data = []
    for temp_img in temp_imges:
              # Blur the image to improve performance
              temp = cv2.GaussianBlur(temp_img, (3, 3), 0)
              # Use sobel edge detection
              temp = cv2.Canny(temp, threshold1=4, threshold2=100)
              # flatten the data to be prepared
              flat_data.append(temp.flatten())
                 
    #convert X_train to numpy
    X_train = np.array(flat_data)
    #convert Y_train to numpy
    labels = np.array(labels)
    #Normalize pixels
    X_train = X_train.astype('float32')/255.0
    Y_train = labels
    print()
    print('Loaded', len(X_train), 'images for training,', 'Train data shape =', X_train.shape)
    return X_train,Y_train

def load_train_data_RGB():
    size = 90, 90
    images=[]
    labels = []
    images_per_folder=0
    print("LOADING DATA FROM : ", end="")
    for folder in os.listdir(train_dir):
        print(folder, end=' | ')
        for image in os.listdir(train_dir + "/" + folder):
          if images_per_folder == 1000:
            images_per_folder=0
            break;
          # read image
          temp_img = cv2.imread(train_dir + '/' + folder + '/' + image)
          # resize image
          temp_img = cv2.resize(temp_img, size)
          images.append(temp_img.flatten())
          labels.append(labels_dict[folder]) 
          images_per_folder+=1 
    #convert X_train to numpy 
    Y_train = np.array(labels)
    #convert Y_train to numpy
    X_train = np.array(images)
    print()
    print('Loaded', len(X_train), 'images for training,', 'Train data shape =', X_train.shape)
    return X_train,Y_train

def load_train_data_BINARY():
    size = 90, 90
    images=[]
    labels = []
    images_per_folder=0
    print("LOADING DATA FROM : ", end="")
    for folder in os.listdir(train_dir):
        print(folder, end=' | ')
        for image in os.listdir(train_dir + "/" + folder):
          if images_per_folder == 1000:
            images_per_folder=0
            break;
          # read image
          temp_img = cv2.imread(train_dir + '/' + folder + '/' + image,0)
          # resize image
          temp_img = cv2.resize(temp_img, size)
          # Binary classification
          _,threshold = cv2.threshold(temp_img, 149, 255, cv2.THRESH_BINARY)
          images.append(threshold)
          labels.append(labels_dict[folder]) 
          images_per_folder+=1

    return images,labels

def load_train_data_GRAY():
    size = 90, 90
    images=[]
    labels = []
    images_per_folder=0
    print("LOADING DATA FROM : ", end="")
    for folder in os.listdir(train_dir):
        print(folder, end=' | ')
        for image in os.listdir(train_dir + "/" + folder):
          if images_per_folder == 1000:
            images_per_folder=0
            break;
          # read image
          temp_img = cv2.imread(train_dir + '/' + folder + '/' + image,0)
          # resize image
          temp_img = cv2.resize(temp_img, size)
          images.append(temp_img.flatten())
          labels.append(labels_dict[folder]) 
          images_per_folder+=1 
    #convert Y_train to numpy 
    Y_train = np.array(labels)
    #convert X_train to numpy
    X_train = np.array(images)
    #Normalize pixels of X_train
    X_train = X_train.astype('float32')/255.0
    print()
    print('Loaded', len(X_train), 'images for training,', 'Train data shape =', X_train.shape)
    return X_train,Y_train

def load_test_data_RGB():
    labels = []
    flat_data = []
    size = 90, 90
    for image in os.listdir(test_dir):
        # read image
        temp_img = cv2.imread(test_dir + '/'+ image)
        # resize image
        temp_img = cv2.resize(temp_img, size)
        # flatten the data to be prepared
        flat_data.append(temp_img.flatten())
        labels.append(labels_dict[image.split('_')[0]])

    #convert X_test to numpy
    X_test = np.array(flat_data)
    #convert Y_test to numpy
    Y_test = np.array(labels)
    print("\n")
    print('Loaded', len(X_test), 'images for testing,', 'Test data shape =', X_test.shape)

    return X_test, Y_test

def load_test_data_GRAY():
    labels = []
    flat_data = []
    size = 90, 90
    for image in os.listdir(test_dir):
        # read image
        temp_img = cv2.imread(test_dir + '/'+ image,0)
        # resize image
        temp_img = cv2.resize(temp_img, size)
        # flatten the data to be prepared
        flat_data.append(temp_img.flatten())
        labels.append(labels_dict[image.split('_')[0]])

    #convert X_test to numpy
    X_test = np.array(flat_data)
    #normalize X_test
    X_test = X_test.astype('float32')/255.0
    #convert Y_test to numpy
    Y_test = np.array(labels)
    print("\n")
    print('Loaded', len(X_test), 'images for testing,', 'Test data shape =', X_test.shape)

    return X_test, Y_test

def load_test_data_BINARY():
    labels = []
    flat_data = []
    size = 90, 90
    for image in os.listdir(test_dir):
        # read image
        temp_img = cv2.imread(test_dir + '/'+ image,0)
        # resize image
        temp_img = cv2.resize(temp_img, size)
        # Binary classification
        _,threshold = cv2.threshold(temp_img, 149, 255, cv2.THRESH_BINARY)
        # flatten the data to be prepared
        flat_data.append(threshold.flatten())
        labels.append(labels_dict[image.split('_')[0]])
    #convert X_Test to numpy
    X_test = np.array(flat_data)
    #convert Y_test to numpy
    Y_test = np.array(labels)
    #normalize pixels of X_test 
    X_test = X_test.astype('float32')/255.0
    print("\n")
    print('Loaded', len(X_test), 'images for testing,', 'Test data shape =', X_test.shape)

    return X_test, Y_test

"""**Load data with RGB**"""

X_train_RGB, Y_train_RGB =  load_train_data_RGB()
X_test_RGB, Y_test_RGB = load_test_data_RGB()

"""**Load data with Gray**"""

X_train_GRAY, Y_train_GRAY =  load_train_data_GRAY()
X_test_GRAY, Y_test_GRAY = load_test_data_GRAY()

"""**Load data with Binary**"""

X_train_BINARY, Y_train_BINARY =  load_train_data_BINARY()
X_train_BINARY, Y_train_BINARY = imageProcessing(X_train_BINARY, Y_train_BINARY)
X_test_BINARY, Y_test_BINARY = load_test_data_BINARY()

"""**Functions of Classifiers**"""

def SVM(X_train,Y_train):
    model = SVC(kernel='linear')
    model.fit(X_train, Y_train)
    return model

def Naive_Base(X_train,Y_train):
    gnb = GaussianNB()
    model = gnb.fit(X_train, Y_train)
    return model

def KNN(X_train,Y_train):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, Y_train)
    return model

def DecisionTree(X_train,Y_train):
    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train, Y_train)
    return clf
    
def MLP(X_train,Y_train):
    mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
    mlp.fit(X_train, Y_train)
    return mlp

def logisticRegression(X_train,Y_train):
  lR = LogisticRegression()
  lR.fit(X_train,Y_train)
  return lR

"""**Train RGB X_train using Decision Tree**"""

#call model on training data 
model = DecisionTree(X_train_RGB, Y_train_RGB)
Y_pred_RGB = model.predict(X_test_RGB)

"""**Calculate Accuracy, Precision and Recall**"""

# calculate accuracy
accuracy = accuracy_score(Y_test_RGB, Y_pred_RGB)
print('Model accuracy is: ', accuracy)

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:", precision_score(Y_test_RGB, Y_pred_RGB, average='micro'))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:", recall_score(Y_test_RGB, Y_pred_RGB, average='micro'))

"""**Train Gray X_train using KNN**"""

model = KNN(X_train_GRAY, Y_train_GRAY)
Y_pred_GRAY = model.predict(X_test_GRAY)

"""**Calculate Accuracy, Precision and Recall**"""

# calculate accuracy
accuracy = accuracy_score(Y_test_GRAY, Y_pred_GRAY)
print('Model accuracy is: ', accuracy)

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:", precision_score(Y_test_GRAY, Y_pred_GRAY, average='micro'))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:", recall_score(Y_test_GRAY, Y_pred_GRAY, average='micro'))

"""**Train Binary X_train using MLP**"""

model = MLP(X_train_BINARY,Y_train_BINARY)
Y_pred_BINARY = model.predict(X_test_BINARY)

"""**Calculate Accuracy, Precision and Recall**"""

# calculate accuracy
accuracy = accuracy_score(Y_test_BINARY, Y_pred_BINARY)
print('Model accuracy is: ', accuracy)

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:", precision_score(Y_test_BINARY, Y_pred_BINARY, average='micro'))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:", recall_score(Y_test_BINARY, Y_pred_BINARY, average='micro'))