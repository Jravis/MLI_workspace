"""
In the absence of traing data one can use transfer learning model InceptionV3 (any other)
to extract features from images and use those features for clustering.
"""

import tensorflow as tf
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# from tensorflow.keras.preprocessing.image import img_to_array
import cv2 as cv
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment as linear_assignment
import pandas as pd
import numpy as np
from tqdm import tqdm
import shutil
import os




def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    
    y_true = np.int64(y_true)
    assert y_pred.size == y_true.size
    
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)

    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def get_features(src_dir, trans_learn_model): 
    """_summary_

    Args:
        src_dir (_type_): _description_
        trans_learn_model (_type_): _description_

    Raises:
        Exception: _description_
    """
   
    if trans_learn_model=='MobileNet':
        
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    elif trans_learn_model=='InceptionV3':
        from tensorflow.keras.applications.inception_v3 import preprocess_input
        model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    else:
        raise Exception("Give valid name for transfer learning model")

    feature_list = []
    img_name = []
    count = 0
    for file in tqdm(os.listdir(src_dir)):
        img  = cv.resize(cv.imread(src_dir+file), (224, 224), cv.INTER_AREA)
        if count > 5000:
            break
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x).flatten()
        feature_list.append(features)
        tmp  = np.array(feature_list)
        print("size in bytes", tmp.nbytes)
        img_name.append(file)
        count+=1
    return feature_list, img_name

src_dir = r"C:\Users\\lenovo\\image_reco_workspace\\packages\\data\\cats-dogs_train\\"

img_features, img_label = get_features(src_dir, 'MobileNet')
k = 2
clusters = KMeans(k, random_state = 40)
clusters.fit(img_features)

image_cluster = pd.DataFrame(img_label, columns=['image'])
image_cluster["clusterid"] = clusters.labels_

print(image_cluster.head(10))

y_true = [1 if name =='cat' else 0 for name in img_label]

acc = cluster_acc(y_true, clusters.labels_)
print("Accuracy = ", (acc * 100.0), "%")


# Segregate the images in respective folders

if not os.path.isdir(src_dir+'cats'):
    os.makedirs(src_dir + 'cats')

if not os.path.isdir(src_dir+'dogs'):
    os.makedirs(src_dir + 'dogs')

for i in range(len(image_cluster)):
    if image_cluster['clusterid'][i]==1:
        shutil.move(src_dir+image_cluster['image'][i], src_dir+'cats\\'+image_cluster['image'][i])
    else:
        shutil.move(src_dir+image_cluster['image'][i], src_dir+'dogs\\'+image_cluster['image'][i])


# get test data feature
# Accuracy  = (TP+TN)/(TP + FP + TN + FN)

