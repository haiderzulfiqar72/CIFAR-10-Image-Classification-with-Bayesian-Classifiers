import pickle
import numpy as np
import matplotlib.pyplot as plt
from random import random
from tqdm import tqdm
import pandas as pd
from skimage.transform import resize
from scipy.stats import norm
from scipy import stats
from scipy.stats import multivariate_normal
from tqdm import tqdm

def class_acc(pred, gt):
    count=0   #number of misclassified samples
       
    for i_index, i in enumerate(gt):
        if pred[i_index]!= gt[i_index]:
            count= count + 1   
    accuracy= 1 - count/len(gt)
    return accuracy

def cifar10_color(X):   
    Xp = X
    return resize(Xp, (50000, 1, 1, 3), preserve_range=True).reshape(50000, 3)


def cifar10_naivebayes_learn(Xp, Y):
    class_label ={}
    mu= {}
    sigma= {}
    p={}
    
    for i, label in enumerate(Y):
        if label in class_label:
            class_label[label].append(i)
        else:
            class_label[label] = [i]
    
    for j, k in class_label.items():
        class_label[j] = Xp[k]

    for a, b in class_label.items():
           mu[a]= np.mean(class_label[a],axis=0)
           sigma[a]= np.std(class_label[a],axis=0)
           p[a]= len(class_label[a]) 
    
    prior= np.array(list(p.values())) / sum(p.values())
    
    return class_label, mu, sigma, prior 

def cifar10_classifier_naivebayes(X_testb, mu, sigma, prior):
    
    normal_dist={}
    every_class = [None]*10
    for i, j in enumerate(X_testb):
        
        for k, l in enumerate(mu):
            every_class[l]= np.prod(norm.pdf(X_testb[i], list(mu.values())[k], list(sigma.values())[k])) * prior[k]
         
        normal_dist[i]= np.argmax(every_class)
        
    return every_class, normal_dist


#part2

def cifar_10_bayes_learn(Xf,Y):
    class_label ={}
    mu= {}
    sigma= {}
    p={}
    
    for i, label in enumerate(Y):
        if label in class_label:
            class_label[label].append(i)
        else:
            class_label[label] = [i]
    
    for j, k in class_label.items():
        class_label[j] = Xf[k]

    for a, b in class_label.items():
           mu[a]= np.mean(class_label[a],axis=0)
           sigma[a]= np.cov(class_label[a],rowvar= False)
           p[a]= len(class_label[a]) 
    
    prior= np.array(list(p.values())) / sum(p.values())
    
    return class_label, mu, sigma, prior 
    

def cifar10_classifier_bayes(x,mu,sigma,p):
    normal_dist={}
    every_class = [None]*10
    for i, j in enumerate(x):
        for k, l in enumerate(mu):
            every_class[l]= multivariate_normal.pdf(x[i], list(mu.values())[k], list(sigma.values())[k]) * p[k]
         
        normal_dist[i]= np.argmax(every_class)
        
    return every_class, normal_dist

#part3

def cifar10_2x2_color(X): 
    
    Xr = X
    return resize(Xr, (50000, 2, 2, 3), preserve_range=True).reshape(50000, 12)

def cifar10_CxC_color(X,C):
    Xr = X
    return resize(Xr, (50000, C, C, 3), preserve_range=True).reshape(50000, 3*C**2)
    
        
def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

datadict = unpickle('E:/Study Material/Masters/Studies/Semester 1/Introduction to Pattern Recognition and Machine Learning - DATA.ML.100/Excercises/Excercise Week 3/cifar-10-batches-py/data_batch_1')
X = datadict["data"]
Y = datadict["labels"]

datadict_1 = unpickle('E:/Study Material/Masters/Studies\Semester 1/Introduction to Pattern Recognition and Machine Learning - DATA.ML.100/Excercises/Excercise Week 3/cifar-10-batches-py/data_batch_2')
X_1 = datadict_1["data"]
Y_1 = datadict_1["labels"]

datadict_2 = unpickle('E:/Study Material/Masters/Studies\Semester 1/Introduction to Pattern Recognition and Machine Learning - DATA.ML.100/Excercises/Excercise Week 3/cifar-10-batches-py/data_batch_3')
X_2 = datadict_2["data"]
Y_2 = datadict_2["labels"]

datadict_3 = unpickle('E:/Study Material/Masters/Studies\Semester 1/Introduction to Pattern Recognition and Machine Learning - DATA.ML.100/Excercises/Excercise Week 3/cifar-10-batches-py/data_batch_4')
X_3 = datadict_3["data"]
Y_3 = datadict_3["labels"]

datadict_4 = unpickle('E:/Study Material/Masters/Studies\Semester 1/Introduction to Pattern Recognition and Machine Learning - DATA.ML.100/Excercises/Excercise Week 3/cifar-10-batches-py/data_batch_5')
X_4 = datadict_4["data"]
Y_4 = datadict_4["labels"]

datadict_tb = unpickle('E:/Study Material/Masters/Studies\Semester 1/Introduction to Pattern Recognition and Machine Learning - DATA.ML.100/Excercises/Excercise Week 3/cifar-10-batches-py/test_batch')
X_tb = datadict_tb["data"]
Y_tb = datadict_tb["labels"]

x_merge= np.concatenate([X, X_1, X_2, X_3, X_4])
y_merge= np.concatenate([Y, Y_1, Y_2, Y_3, Y_4])

del X, X_1, X_2, X_3, X_4, Y, Y_1, Y_2, Y_3, Y_4

labeldict = unpickle('E:/Study Material/Masters/Studies/Semester 1/Introduction to Pattern Recognition and Machine Learning - DATA.ML.100/Excercises/Excercise Week 3/cifar-10-batches-py/batches.meta')
label_names = labeldict["label_names"]

X = x_merge.reshape(50000, 3, 32, 32).transpose(0,2,3,1).astype("uint8").astype('int')
Y = np.array(y_merge)

X_tb= X_tb.reshape(10000,3,32,32).transpose(0,2,3,1).astype("uint8").astype('int')
X_testb= resize(X_tb, (10000, 1, 1, 3), preserve_range=True).reshape(10000, 3)
X_testb1= resize(X_tb, (10000, 2, 2, 3), preserve_range=True).reshape(10000, 12)
Y_tb = np.array(Y_tb)

print(X.shape)

for i in range(X.shape[0]):
    # Show some images randomly
    if random() > 0.999999999:
        plt.figure(1);
        plt.clf()
        plt.imshow(X[i])
        plt.title(f"Image {i} label={label_names[Y[i]]} (num {Y[i]})")
        plt.pause(1)


Xp= cifar10_color(X)

class_label, mu, sigma, prior = cifar10_naivebayes_learn(Xp, Y)
every_class, normal_dist = cifar10_classifier_naivebayes(X_testb, mu, sigma, prior)
z_1= class_acc(list(normal_dist.values()), Y_tb)
print(f'\nAccuracy Function For Naive Bayes:{round(z_1*100,2)}%')

# class_label, mu, sigma, prior = cifar_10_bayes_learn(Xp, Y)
# every_class, normal_dist = cifar10_classifier_bayes(X_testb, mu, sigma, prior)

# z_2= class_acc(list(normal_dist.values()), Y_tb)
# print(f'\nAccuracy Function For Multivariate Normal:{round(z_2*100,2)}%')

# Xr= cifar10_2x2_color(X)
# class_label, mu, sigma, prior = cifar_10_bayes_learn(Xr, Y)
# every_class, normal_dist = cifar10_classifier_bayes(X_testb1, mu, sigma, prior)

# # z_3= class_acc(list(normal_dist.values()), Y_tb)
# # print(f'\nAccuracy Function For  :{round(z_3*100,2)}%')

sizes = [1, 2, 4, 8, 12, 14, 16] #for size 32, its taking a lot of time to run

arr= []
for j in tqdm(range(len(sizes))):
    i = sizes[j]
    xri = cifar10_CxC_color(X,i)
    X_testb2= resize(X_tb, (10000, i, i, 3), preserve_range=True).reshape(10000, 3* i**2)
    
    # class_label, mu, sigma, prior = cifar10_naivebayes_learn(Xr, Y)
    # every_class, normal_dist = cifar10_classifier_naivebayes(X_testb2, mu, sigma, prior)
    
    class_label, mu, sigma, prior= cifar_10_bayes_learn(xri, Y)
    every_class, normal_dist= cifar10_classifier_bayes(X_testb2, mu, sigma, prior)
    
    z_4= class_acc(list(normal_dist.values()), Y_tb)
    arr.append(z_4)
    print(f'\nAccuracy Function For {i}*{i}*3 :{round(z_4*100,2)}%')

plt.plot(sizes, arr, label= 'Bayesian Classifier')
plt.legend()
plt.show()

