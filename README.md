# ASL-Classifiers
> American Sign Language (ASL) is a natural language that serves as the predominant sign language of Deaf communities in the United States and most of Anglophone Canada. 

> ASL is a complete and organized visual language that is expressed by facial expression as well as movements and motions with the hands.

> Machine learning models build on different types of data :
 - RGB
 - BINARY
 - GRAY
 
> Classifiers : 
 - SVM
 - MLP
 - KNN

# Data
>The data set is a collection of images of alphabets from the American Sign Language, separated in 29 folders which represent the various classes.

> The ASL Alphabet data set provides 87,000 images of the ASL alphabet.

> The test data set contains a mere 29 images, to encourage the use of real-world test images.

> There are 2 data sets utilized in this notebook:
 - *ASL Alphabet train* - This data set is the basis for the model.
 - *ASL Alphabet Test* - This data set was made specifically for validating the model created using the above data set, and is intended to be used to improve the feature engineering and modeling process to make it more versatile in "the wild" with less contrived images.

> It is available on Kaggle as the ASL Alphabet Dataset. https://www.kaggle.com/grassknoted/asl-alphabet.

# Functions 
> *imageProcessing*
```
function to load and process BINARY images.
```
> *load_train_data_RGB*
```
function to load train data in RGB type.
```
> *load_train_data_GRAY*
```
function to load train data in GRAY type.
```
> *load_train_data_BINARY*
```
function to load train data in BINARY type.
```
> *load_test_data_RGB*
```
function to load test data in RGB type.
```
> *load_test_data_GRAY*
```
function to load test data in GRAY type.
```
> *load_test_data_BINARY*
```
function to load test data in BINARY type.
```
> *SVM , KNN and MLP*
```
function to build the classifiers.
```
*Note*
> According to the weakness of images processing the accuracy of the models is weak which will be better in the CNN model.
