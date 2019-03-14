import os, cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

model = load_model('model_cactus.h5')

directory = '/home/sarvesh/ML_Github/cactus/'
imgtype = '/*.jpg'

test = sorted([x for x in os.listdir(directory + 'test' + '/') if x.endswith(imgtype[2:])])

def preprocess_image(image):
    
    #read in a grayscale image
    img = cv2.imread(directory + 'test' + '/' + image)
    
    #convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #apply binary threshold
    ret, thresh_bin = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)
    
    return gray

X_test = []

for image in test:
    img = preprocess_image(image)
    X_test.append(img)

X_test = np.array(X_test).reshape(-1, 32, 32, 1)

y_test = model.predict(X_test)

y_test = y_test.squeeze()

df_test = pd.DataFrame()
df_test['id'] = np.array(test).T
df_test['has_cactus'] = np.array(y_test).T

df_test.to_csv('solution.csv', index = False)