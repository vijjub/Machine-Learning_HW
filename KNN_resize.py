__author__ ="Vijendra Kumar Bhogadi"
import os, time, sys
from PIL import Image
from PIL import Image as im
import PIL

# Linear Algebra Library
from numpy import *
from numpy.linalg import *
import numpy as np
import numpy.linalg as la

import math
#from cvxopt import *
import cvxopt
import cvxopt.solvers


##Training data divided into 5 parts equally
path_part_1 = "/home/vijjub/Documents/Machine Learning/Homework 2/t1"
path_part_2  = "/home/vijjub/Documents/Machine Learning/Homework 2/t2"
path_part_3 = "/home/vijjub/Documents/Machine Learning/Homework 2/t3"
path_part_4  = "/home/vijjub/Documents/Machine Learning/Homework 2/t4"
path_part_5  = "/home/vijjub/Documents/Machine Learning/Homework 2/t5"

##Trainig and test data for all iterations
iteration_1 = [path_part_1,path_part_2,path_part_3,path_part_4]
test_1 = [path_part_5]

iteration_2 = [path_part_1,path_part_2,path_part_3,path_part_5]
test_2 = [path_part_4]

iteration_3 = [path_part_1,path_part_2,path_part_5,path_part_4]
test_3 = [path_part_3]

iteration_4 = [path_part_1,path_part_5,path_part_3,path_part_4]
test_4 = [path_part_2]

iteration_5 = [path_part_5,path_part_2,path_part_3,path_part_4]
test_5 = [path_part_1]

print "Every image has been reduced to 56*46 size"

class Plotter():
    def __init__(self, Name ,val = None, ylabel = None):
        self.name   = Name
        self.values = val
        self.ylabel = ylabel
        
    def plot(self, values, label):
        plt.plot(values)
        plt.ylabel(label)
        plt.show()



class PCA:
    def __init__(self):

        self.X                = np.reshape(([0] * 2576), [-1,1])
        self.eigen_val        = None
        self.eigen_vect      = None
        self.new_eigen_val    = None
        self.new_eigen_vect   = None
        self.prod_val         = None
        self.X_mean           = [0] * 2
        self.XT               = None



    def mod_X(self, vect):
        self.X = np.append(self.X, vect, axis=1)

    def form_X(self):
        self.X = self.X[0:,1:]

    def subtract_mean_X(self):
        for each in range(len(self.X)):
            self.X[each,:] -= float(float(self.X[each, :].sum())/len(self.X[0,:]))

    def transpose_X(self):
        self.XT = np.transpose(self.X)

    def X_T_prod_X(self):
        self.prod = np.dot(self.XT,self.X)

    def compute_eig_val_vect(self):
        self.eigen_val , self.eigen_vect = np.linalg.eig(self.prod)

    def sorting_eigen_vals(self):
        self.index = np.argsort(-self.eigen_val)


    def arrange_vectors(self):
        self.new_eig_val  = self.eigen_val[self.index]
        self.new_eig_vect = self.eigen_vect[:,self.index]


    def multipy_eigen_vectors_by_X(self):
        self.mult_val = np.dot(self.X, self.new_eig_vect)

    def normalization(self):
        for each in range (len(self.mult_val[0])):
            self.mult_val[:,each] /= la.norm(self.mult_val[:,each])



    def get_mean_face_X(self):
        return self.mean_X

    def normal_T(self):
        self.u = np.transpose(self.mult_val)


    def top_k_eigen_vectors(self):
        self.p = self.u[:50,]


    def get_X(self):
        return self.X


    def get_p(self):
        return self.p




def Train(path):
    pca_train = PCA()
    labels = []
    for path in iteration_1:
           for each in os.listdir(path):
               for each1 in os.listdir(path + '/' + each):
                   labels.append(each)
                   a_img = Image.open(path + '/' + each + '/' + each1)
                   im = a_img.resize((56, 46), PIL.Image.ANTIALIAS)
                   get_image = im.getdata()
                   column_image = np.reshape(get_image, [-1,1])
                   pca_train.mod_X(column_image)

    pca_train.form_X()
    pca_train.subtract_mean_X()
    pca_train.transpose_X()
    pca_train.X_T_prod_X()
    pca_train.compute_eig_val_vect()
    pca_train.sorting_eigen_vals()
    pca_train.arrange_vectors()
    pca_train.multipy_eigen_vectors_by_X()
    pca_train.normalization()
    pca_train.normal_T()
    pca_train.top_k_eigen_vectors()
    p = pca_train.get_p()
    print "shape of matrix p "+str(p.shape)
    ##y is the final data matrix
    y = np.dot(p,pca_train.get_X())
    print "Final dimensions of the training set projected on eigen space "+str(y.shape)
    return y,p,labels

def Testing_KNN(test_path,y,p,train_names):
    print "started KNN for One NN"
    pca_test = PCA()
    actual_names = []
    for path in test_path:
           for each in os.listdir(path):
               for each1 in os.listdir(path + '/' + each):
                   actual_names.append(each)
                   a_img = Image.open(path + '/' + each + '/' + each1)
                   im = a_img.resize((56, 46), PIL.Image.ANTIALIAS)
                   get_image = im.getdata()
                   column_image = np.reshape(get_image, [-1,1])
                   pca_test.mod_X(column_image)

    pca_test.form_X()
    pca_test.subtract_mean_X()
    X = pca_test.get_X()

    ##get mean face of testdata
    print X.shape
    print p.shape


    ##multiplying with p to project on the reduced space
    
    F = np.dot(p,X)
    print F.shape
    train_data = y
    test_data = F
    match = 0
    mismatch = 0

    for each in range((test_data.shape[1])):
        lis = []
        for each1 in range(train_data.shape[1]):
            lis.append((la.norm(test_data[:,each] - train_data[:,each1])))

        x = lis.index(min(lis))
        print ('Actual: ' + actual_names[each] + ' Predicted: '+ train_names[x])
        if (actual_names[each] == train_names[x]):
            match += 1
        else:
            mismatch += 1

    accuracy = float(match*100/(mismatch + match))

     
    print 'The accuracy of the Classificaiton: ' + str(accuracy) + ' %'
    print 'Wrong Predictions  : ' + str(match)
    print 'Correct Predictions: ' + str(mismatch)
    return accuracy
   


    
    

start = time.time()
print"First Iteration"
print "==========================================================================\n"
y,p,train_names = Train(iteration_1)
acc1 = Testing_KNN(test_1, y,p,train_names)
print "==========================================================================\n"
print"Second Iteration"
y,p,train_names = Train(iteration_2)
acc2 = Testing_KNN(test_2, y,p,train_names)
print "==========================================================================\n"
print"Third Iteration"
y,p,train_names = Train(iteration_3)
acc3 = Testing_KNN(test_3, y,p,train_names)
print "==========================================================================\n"
print"Fourth Iteration"
y,p,train_names = Train(iteration_4)
acc4 = Testing_KNN(test_4, y,p,train_names)
print "==========================================================================\n"
print"Fifth Iteration"
y,p,train_names = Train(iteration_5)
acc5 = Testing_KNN(test_5, y,p,train_names)
print "==========================================================================\n"
total_acc = float((acc1+acc2+acc3+acc4+acc5)/5.0)
print "Final 5-fold accuracy is "+str(total_acc)
print "==========================================================================\n"
end = time.time()
print 'time taken = ' + str((end - start)/60) + ' minutes'

print "==========================================================================\n"









