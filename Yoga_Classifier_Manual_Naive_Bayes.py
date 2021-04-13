# In[1]:


# This function prepare the data by reading it from a file and converting it into a useful format for training and testing
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def preprocess(loc):
    
    # Read train.csv and turn all string nuric to float numeric
    all_data = []
    labels = []
    imputer = KNNImputer(n_neighbors=100)
    read_to_arrays(loc, labels, all_data)
        
    
    # Turning data into a numpy array object.
    all_data = np.array(all_data)
    # Imputing the dataset using KNN nearest neighbours.
    all_data = imputer.fit_transform(all_data)
#     print(all_data)
    return all_data, labels


def read_to_arrays(loc, labels, all_data):
    with open(loc, mode='r') as fin:
        for line in fin:
            attr = line.strip().split(",")
            # Segregating the labels from the dataset.
            labels.append(attr[0])
            attr = attr[1:]
    # 
            # Transforming strings into float
            for x in range(0,len(attr)):
                if (attr[x] == '9999'):
                    attr[x] = np.nan
                else:
                    attr[x] = float(attr[x])


            all_data.append(attr)


# In[2]:


# This function calculat prior probabilities and likelihoods from the training data and using
# them to build a naive Bayes model
def train(data_train, labels_train):
    n_samples, n_attr = data_train.shape
    all_classes = np.unique(labels_train)
    
    # initialise priors, mean and variance.
    priors_p = np.zeros(len(all_classes), dtype=np.float64)
    mean = np.zeros((len(all_classes), n_attr), dtype=np.float64)
    stdev = np.zeros((len(all_classes), n_attr), dtype=np.float64)
    
    
    
    for c in range(0, len(all_classes)):
        data_c = []
        for c2 in range(0,len(labels_train)):
            if all_classes[c] == labels_train[c2]:
                data_c_instance = data_train[c2]
                # Data_c will contain all instances of 1 specific class.
                data_c.append(data_c_instance)
        
        # Now we have data_c which has stat for 1 specific class
        # Calculate the mean and var for each attribute of this class.
        data_c = np.array(data_c)
        mean[c,:] = data_c.mean(axis=0)
        stdev[c,:] = data_c.std(axis=0)
        # Calculate prior for each class, data_c.shape[0] is the frequency of this class occuring in the whole set.
        priors_p[c] = data_c.shape[0] / float(n_samples)
        
    
    
    # Now we have mean, var of each attribute for each class, as well as prior prob for each class.
    return priors_p, mean, stdev, all_classes


# In[3]:


import math
# This function predict classes for new items in a test dataset
def predict(mean, stdev, priors, all_classes):
    
    # Pre-process the testing data.
    data_test, labels_test = preprocess('test.csv')
    class_predict = []
    
    # Loop through each instance and calculate the likelihood of that instance on each class.
    for inst_index ,instance in enumerate(data_test):
        gnb_score_arr = np.zeros(len(all_classes), dtype=np.float64)
        
        for index, c in enumerate(all_classes):
            # Work out log of priors first for this current class.
            priors_log = np.log(priors[index])

            # Pass instance, mean[index], stdev[index] to likelihood function.
            gnb_score = cal_likelihood(instance, mean[index], stdev[index]) + priors_log
            gnb_score_arr[index] = gnb_score

        # Find the max score and what class it belongs to via the index.
        max_gnb = np.max(gnb_score_arr)
        max_index = -1;
        for i, score in enumerate(gnb_score_arr):
            if score == max_gnb:
                max_index = i



        class_predict.append(all_classes[max_index])
            
            
    

    return class_predict, labels_test


def cal_likelihood(instance, mean_class, stdev_class):
    product_log_likelihood = 0

    # Loop through each attribute and compute the likelihood
    for index, attr in enumerate(instance):
        mean = mean_class[index]
        stdev = stdev_class[index]
        var = pow(stdev, 2)
        
        # Compute the normal distribution equation.
        # Split into the first and second term of the gaussian distribution equation.
        first = 1/(np.sqrt(2 * np.pi * var))
        
        # We do not take log() here since log and exponential cancels out.
        second = -(attr - mean)**2 / (2 * var)
        product_log_likelihood += np.log(first) + second
        
        
        
    
    return product_log_likelihood


# In[4]:


# This function evaluate the prediction performance by comparing your model’s class outputs to ground
# truth labels

def evaluate(y_predict, y_test):
    acc_count = 0
    
    # Finidng the total number of correct class Labels.
    for counter in range(0, len(y_predict)):
        if y_predict[counter] == y_test[counter]:
            acc_count+= 1
        
    # Calculating percentage.
    acc = (acc_count/len(y_test)) * 100
        
    return acc


data, labels = preprocess('train.csv')
# Representing the data in a table. 
# print(dataFrame) to see data after preprocessing.
dataFrame = pd.DataFrame(data=data, columns = ["x1","x2","x3","x4","x5","x6","x7","x8","x9","x10","x11"
                                               ,"y1","y2","y3","y4","y5","y6","y7","y8","y9","y10","y11"])
priors, mean, stdev, all_classes = train(data, labels)
y_predict, y_test = predict(mean, stdev, priors, all_classes)
acc = evaluate(y_predict, y_test)

print("################################################################\n")
print("Naive Bayes accuraccy: ", acc)
print("\n################################################################\n")


# In[7]:


# Evaluating the model

# Confusion Matrix
cx = confusion_matrix(y_test, y_predict)

# True positives are the cells along the diagonal.
TP = np.diag(cx)

# False Positive is define here as the sum of each column without the TP case.
FP = cx.sum(axis=0) - TP

# False negative is define here as the sum of each row without the TP case.
FN = cx.sum(axis=1) - TP

# This is define as the sum of all cells that the x and y axis of TP do not touch. AKA everything else that is not TP,FP,FN.
TN = cx.sum() - (FP + FN + TP)


# Initialise precision, recall and F1 arrays, each cell correspond to a class.
precision = np.zeros(len(TP))
recall = np.zeros(len(TP))
F1 = np.zeros(len(TP))

# Calculate precision, recall and F1 for each class using formula given in lecture.
sum_TP = 0
sum_TP_FP = 0
sum_TP_FN = 0
for i in range(0, len(TP)):
    sum_TP += TP[i]
    sum_TP_FP += TP[i] + FP[i]
    sum_TP_FN += TP[i] + FN[i]
    precision[i] = TP[i] / (TP[i] + FP[i])
    recall[i] = TP[i] / (TP[i] + FN[i])
    F1[i] = 2 * (precision[i] * recall[i] / (precision[i] + recall[i]))
    

    
# Calculate micro precision and micro_recall, they have the same value?
micro_precision = sum_TP / sum_TP_FP
micro_recall = sum_TP / sum_TP_FN
micro_F1 = 2 * (micro_precision * micro_recall / (micro_precision + micro_recall))


# Calculate macro precision, recall
sum_precision = 0
sum_recall = 0
for i in range(0, len(precision)):
    sum_precision += precision[i]
    sum_recall += recall[i]
    
macro_precision = sum_precision / len(precision)
macro_recall = sum_recall / len(recall)
macro_F1 = 2 * (macro_precision * macro_recall / (macro_precision + macro_recall))


# Calculate weighted average:
precision_sum = 0
recall_sum = 0
w_top_prec = 0
w_top_recall = 0

freq_dict = {}
for i in y_test:
    if i not in freq_dict:
        freq_dict[i] = 0
    else:
        freq_dict[i] += 1
        
freq = []
for i in freq_dict.values():
    freq.append(i)


for i in range(0,len(precision)):
    precision_sum += precision[i]
    recall_sum += recall[i]
    w_top_prec += precision[i] * freq[i]
    w_top_recall += recall[i] * freq[i]

weighted_average_precision = w_top_prec / len(y_test)
weighted_average_recall = w_top_recall / len(y_test)



# Using micro precision, recall, and f1 score, we found that all 3 scores are the same.
# This is because in a multiclass classification, FN = FP, hence recall = precision, hence F1 = precision = recall.
print("micro precision:", micro_precision, " micro recall:", micro_recall, " micro F1:", micro_F1)
print("macro precision:", macro_precision, " macro recall:", macro_recall, " macro F1:", macro_F1)
print("weighted_average_precision:", weighted_average_precision, " weighted_average_recall:", weighted_average_recall)
print("\n################################################################\n")



# ### Q6
# Engineer your own pose features from the provided keypoints. Instead of using the (x,y) positions of keypoints, you might consider the angles of the limbs or body, or the distances between pairs of keypoints. How does a naive Bayes classifier based on your engineered features compare to the classifier using (x,y) values? Please note that we are interested in explainable features for pose recognition, so simply putting the (x,y) values in a neural network or similar to get an arbitrary embedding will not receive full credit for this question. You should be able to explain the rationale behind your proposed features. Also, don't forget the conditional independence assumption of naive Bayes when proposing new features -- a large set of highly-correlated features may not work well.

# In[6]:

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_selection import SelectKBest, chi2
from matplotlib.backends.backend_pdf import PdfPages



# Read and preprocess train data for feature engineering.
data_6, labels_6 = preprocess('train.csv')
#Perform feature engineering
def feature_engineer(data_6, labels_6):
    new_data = []
    for i in range(0,len(data_6)):
        dist = []

        for j in range (0, 11):
            coords_pair = []
            x = data_6[i][j]
            y = data_6[i][j + 10]
            coords_pair.append(x)
            coords_pair.append(y)
            dist.append(coords_pair)
        # sklearn euclidean distance calculates the distance of each pair of point to each other pair of points.
        # We are going to have 121 (11x11) attributes for each feature once the algorithm finish populating.
        euclid_dist = euclidean_distances(dist, dist)

        inst = []
        for instance in euclid_dist:
            for distance in instance:
                inst.append(distance)

        new_data.append(inst)


    # Now we have a 121x747 matrix base on the original dataset. Very large...
    # Perform feature selection on this data.
    # We select k to be 22 for feature selection, keeping the number of features the same as original.
    new_data = SelectKBest(chi2, k=22).fit_transform(new_data, labels_6)
    
    
    return new_data


# Perform feature engineering
new_data_6 = feature_engineer(data_6, labels_6)
priors_6, mean_6, stdev_6, all_classes_6 = train(new_data_6, labels_6)

# Representing the data in a table
df = pd.DataFrame(data=new_data_6, columns = ["feature1_x2","feature2_x2","feature3_x2","feature4_x2","feature5_x2"
                                              ,"feature6_x2","feature7_x2","feature8_x2","feature9_x2","feature10_x2"
                                              ,"feature11_x2","feature12_x2","feature13_x2","feature14_x2","feature15_x2"
                                              ,"feature16_x2","feature17_x2","feature18_x2","feature19_x2","feature20_x2"
                                              ,"feature21_x2","feature22_x2"])




# Predict_6 is similar to original predict function
# However it transform the coords of the test cases into euclidean distances as well in order to test.
def predict_6(mean, stdev, priors, all_classes):
    
    # Pre-process the testing data.
    data_test, labels_test = preprocess('test.csv')
    data_test = feature_engineer(data_test, labels_test)
    
    class_predict = []
    
    # Loop through each instance and calculate the likelihood of that instance on each class.
    for inst_index ,instance in enumerate(data_test):
        gnb_score_arr = np.zeros(len(all_classes), dtype=np.float64)
        
        for index, c in enumerate(all_classes):
            # Work out log of priors first for this current class.
            priors_log = np.log(priors[index])

            # Pass instance, mean[index], stdev[index] to likelihood function.
            gnb_score = cal_likelihood(instance, mean[index], stdev[index]) + priors_log
            gnb_score_arr[index] = gnb_score

        # Find the max score and what class it belongs to via the index.
        max_gnb = np.max(gnb_score_arr)
        max_index = -1;
        for i, score in enumerate(gnb_score_arr):
            if score == max_gnb:
                max_index = i



        class_predict.append(all_classes[max_index])
            
            
    

    return class_predict, labels_test



# The result with feature engineering
y_predict6, y_test6 = predict_6(mean_6, stdev_6, priors_6, all_classes_6)
acc6 = evaluate(y_predict6, y_test6)


# Our accuracy scores.
print("Feature engineering accuracy: ", acc6)
print("\n################################################################\n")

