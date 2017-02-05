__author__ ="Vijendra Kumar Bhogadi"
import os, time, sys
from PIL import Image as im
from numpy import *
from numpy.linalg import *
import numpy as np
import numpy.linalg as la
import math
import cvxopt
import cvxopt.solvers

##Folder locations of the dataset
path_training = "/home/vijjub/Documents/Machine Learning/Homework 2/train"
path_testing  = "/home/vijjub/Documents/Machine Learning/Homework 2/test"

##Using the gaussian kernel
def Gauss(X, m, deviation):
            return math.exp(-la.norm(X-m)/ (2 * math.pow(deviation,2)))
##For Linear Kernel
def linear(X_i, X_j):
    return np.dot(X_i,X_j)

##get support vectors for the classification
def get_support_vectors(Classifier, classes, res):
    index  = res > 1e-0
    mul = np.arange(len(res))[index]
    sup_init = res[index]
    sup_vect = Classifier[index]
    classes = classes.transpose()
    sup_init_y = classes[index]
    return sup_init, sup_vect, sup_init_y, index, mul

def get_bias(sup_init, sup_vect, sup_init_y):
    bias = 0
    for i in range(len(sup_init)):
        bias+= sup_init_y[i]
        summ = 0
        for j in range(len(sup_vect)):
            summ += (sup_init[j] * sup_init_y[j] * Gauss(sup_vect[i], sup_vect[j], 100))
            
        bias = bias - summ
    bias= bias/len(sup_init)
    return bias


## Solving the quadratic problem using cvxopt solver.

def get_train_results(Classifier, classes):
##we need X^T * X for solution. thats why we prepare a grammitian matrix
    points, features = Classifier.shape
    K = np.zeros((points, points))
    for i in range(points):
        for j in range(points):
            K[i][j] = Gauss(Classifier[i], Classifier[j], 100)
         
    X = (np.outer(classes, classes))*K
    P = cvxopt.matrix(X)

    X = (np.ones(points))
    X2 = X*(-1)
   
    q = cvxopt.matrix(X2)

    A = cvxopt.matrix(classes, (1,points))
    b = cvxopt.matrix(0.0)

    G = cvxopt.matrix(np.diag(np.ones(points) * -1))
    h = cvxopt.matrix(np.zeros(points))

    cvxopt.solvers.options['show_progress'] = False
    solver = cvxopt.solvers.qp(P, q, G, h, A, b)
    sol = np.ravel(solver['x'])
    
    sup_init, sup_vect, sup_init_y, index, mul= get_support_vectors(Classifier, classes, sol)
    b = get_bias(sup_init, sup_vect, sup_init_y)

    return sol,sup_init, sup_vect, sup_init_y, index, mul, b

##Projecting vector into other dimesnsions for better classification
def new_image_classify(image, sup_init, sup_init_y ,sup_vect, index, bias):
    vector = np.zeros(len(image))
    summ = 0
    for lag, y_i, v in zip(sup_init, sup_init_y, sup_vect):
            summ += lag * y_i * Gauss(image, v, 100)
    vector = summ
    return (vector + bias)



##labelling image with +1 if it belongs to the same class or other cllasses
def get_classes(each, length):
    classes = np.ones((1,length))
    classes= (classes * -1)
    classes[0, (each * 5)] = 1
    classes[0, ((each * 5) + 1)] = 1
    classes[0, ((each * 5) + 2)] = 1
    classes[0, ((each * 5) + 3)] = 1
    classes[0, ((each * 5) + 4)] = 1
    return classes

class SVM:
    def __init__(self, name):
        self.name           = name
        self.X              = np.reshape(([0] * 10304), [-1,1])
        self.X_T             = None

    def set_name(self, _name):
        self.name = _name

    def get_name(self):
        return (self._name)

    def big_X(self, add):
        self.X = np.append(self.X, add, axis=1)

    def actual_X(self):
        self.X = self.X[0:,1:]
        
    def X_transpose(self):
        self.X_T = np.transpose(self.X)

    
    def get_X(self):
        return self.X

    def get_X_T(self):
        return np.transpose(self.X)



##Training by using one vs All classification method

def Train(path):
    ATT_image_rec = SVM('machine1')
    labels = []
##Reading each image and convering them to an column vector
    for folder in os.listdir(path):
        for files in os.listdir(path + '/' + folder):
            labels.append(folder)
            image = im.open(path + '/' + folder + '/' + files)
            get_image = image.getdata()
            column_image = np.reshape(get_image, [-1,1])
            ATT_image_rec.big_X(column_image)

    ATT_image_rec.actual_X()
    ATT_image_rec.X_transpose()
##For all the classes trying to find the support vectors and b
    machines = {}
    for i in range(40):
        classes = get_classes(i, len(labels))
        final = {}
        
        sol,sup_init, sup_vect, sup_init_y, index, mul, b = get_train_results(ATT_image_rec.get_X_T(),classes)
        final['sol'] = sol
        final['sup_init'] = sup_init
        final['sup_vect'] = sup_vect
        final['sup_init_y'] = sup_init_y
        final['index'] = index
        final['mul'] = mul
        final['b'] = b
        machines[labels[i*5]] = final
        print '\nCompleted Training for class : ' + str(labels[i*5])
        print 'No.of support vectors for this data : ' + str(len(sup_vect))
        print 'Total no.of trained classes: '+str(i+1)
        print 'Please wait........... Training next class......'
        
    return machines


##Testing function with 50% data.We have the support vector machines for each class vs all classes

def Testing(path, machines):
    image_matrix = SVM('test')
    labels = []
##Reading each image and convering them to an column vector
    for folder in os.listdir(path):
        for files in os.listdir(path + '/' + folder):
            labels.append(folder)
            image = im.open(path + '/' + folder + '/' + files)
            get_image = image.getdata()
            vector = np.reshape(get_image, [-1,1])            
            image_matrix.big_X(vector)

    image_matrix.actual_X()
    image_matrix.X_transpose()
    feature_vector= image_matrix.get_X_T()
    class_list = []
    for key in machines.keys():
        class_list.append(key)
    summ = 0
    correct = 0
    wrong =  0
    for i in feature_vector:
        results = []
        peak_val = -999999
        for key in machines:
            value = new_image_classify(i, machines[key]['sup_init'], machines[key]['sup_init_y'], machines[key]['sup_vect'],machines[key]['index'], machines[key]['b'])
            if(value > peak_val ):
                peak_val = value
                new_key = key

       
        print 'Actual Class: ' + str(labels[summ]) +'\t'+ ' Predicted Class: ' + str(new_key)
        if(labels[summ] == new_key):
            correct = correct + 1
        else:
            wrong = wrong+1
        summ = summ + 1

    acc = (float(correct)/200.0)*100
    print 'Total Matches are: ' + str(correct)
    print 'Total Wrong Matches are : ' + str(wrong)
    print 'Accuracy : ' + str(acc)+'%'
    return acc
    
       

print "Started Training with 50% data"
##For data split into 50:50 ratio
machines = Train(path_training)
print "Started testing classifier with other 50% data"
pctg1 = Testing(path_testing, machines)
print "\n\nNow starting training by swapping training and testing data"
##For swapped data from the above 
machines2 = Train(path_testing)
print "Now starting testing with data swapped"
pctg2 = Testing(path_training,machines2)
##Final avg accuracy of the classification
print "calculating avg accuaracy of both " 
final_avg = (float(pctg1+pctg2)/2.0)
print "Total Avg percentage accuracy is :"+str(final_avg)+'%'
