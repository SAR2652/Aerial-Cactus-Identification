#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, cv2, glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Lambda, Flatten, BatchNormalization, Convolution2D , MaxPooling2D
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical


# In[2]:


#set directory
directory = '/home/sarvesh/ML_Github/cactus/'
imgtype = '/*.jpg'
path = os.path.dirname(directory)
folders = os.listdir(path)
folders


# In[3]:


#remove unnecessary files
folders.remove('.ipynb_checkpoints')
folders.remove('cactus.ipynb')
folders.remove('train.csv')
folders.remove('sample_submission.csv')
folders.remove('preprocess.ipynb')
folders.remove('model_cactus_weights.h5')
folders.remove('model_cactus.h5')
folders.remove('cactus.py')
#folders.remove('cactus_logistic.py')
folders


# In[4]:


#generate sorted lists of training and testing image names
test = sorted([x for x in os.listdir(directory + folders[0] + '/') if x.endswith(imgtype[2:])])
train = sorted([x for x in os.listdir(directory + folders[1] + '/') if x.endswith(imgtype[2:])])
print(test[:5])


# In[5]:


df_train = pd.read_csv(directory + 'train.csv')
df_train.head()


# In[6]:


df_train['has_cactus'].value_counts()


# In[7]:


y = df_train['has_cactus'].values.tolist()
y[:5]


# In[8]:


def preprocess_image(image, n):
    
    #read in a grayscale image
    img = cv2.imread(directory + folders[n] + '/' + image)
    
    #convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #apply binary threshold
    ret, thresh_bin = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)
    
    return gray


# In[9]:


#empty list for training
X = []

#extract and preprocess every image 
for image in train:
    img = preprocess_image(image, 1)
    X.append(img)
    
#reshape array
X = np.array(X).reshape(-1, 32, 32, 1)


# In[10]:


#create image data generators for testing and validation
train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

val_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')


# In[11]:


#split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[15]:


nb_train_samples = len(X_train)
nb_validation_samples = len(X_val)
batch_size = 50


# In[13]:


#define neural network model
model = Sequential()
model.add(Convolution2D(128, (3, 3), input_shape = (32, 32, 1), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(rate = 0.2))
model.add(Convolution2D(128, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(rate = 0.2))
#flatten 3d feature maps to 1D
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

#compile the model
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#generate a summary of the model
model.summary()


# In[17]:


train_generator = train_datagen.flow(X_train, y_train, batch_size = batch_size)
validation_generator = val_datagen.flow(X_val, y_val, batch_size = batch_size)


# In[ ]:


history = model.fit_generator(
    train_generator, 
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size
)

model.save_weights('model_cactus_weights.h5')
model.save('model_cactus.h5')

#same operations on test data
X_test = []

for image in test:
    img = preprocess_image(image, 0)
    X_test.append(img)

X_test = np.array(X_test).reshape(-1, 32, 32, 1)

y_test = model.predict(X_test)

df_test = pd.DataFrame()
df_test['id'] = np.array(test).T
df_test['has_cactus'] = np.array(y_test).T

df_test.to_csv('solution.csv', index = False)



