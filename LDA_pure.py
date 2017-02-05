
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

import matplotlib.pyplot as plt


##Data files
path_part_1 = "/home/vijjub/Documents/Machine Learning/Homework 2/t1"
path_part_2  = "/home/vijjub/Documents/Machine Learning/Homework 2/t2"
path_part_3 = "/home/vijjub/Documents/Machine Learning/Homework 2/t3"
path_part_4  = "/home/vijjub/Documents/Machine Learning/Homework 2/t4"
path_part_5  = "/home/vijjub/Documents/Machine Learning/Homework 2/t5"

##iterations for 5-fold

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



class Plotter():
    def __init__(self, Name ,val = None, ylabel = None):
        self.name   = Name
        self.values = val
        self.ylabel = ylabel
        
    def plot(self, values, label):
        plt.plot(values)
        plt.ylabel(label)
        plt.show()


class LDA:
    def __init__(self):
        self.X              = np.reshape(([0] * 2576), [-1,1])

    def modify_X(self, add):
        self.X = np.append(self.X, add, axis=1)

    def actual_X(self):
        self.X = self.X[0:,1:]
    
    def mean_face_X(self):
        for each in range(len(self.X)):
            self.X[each,:] -= float(float(self.X[each, :].sum())/len(self.X[0,:]))

    def get_X(self):
        return self.X





def Train(path):
    LDA_train = LDA()
    names = []
    for path in iteration_1:
           for each in os.listdir(path):
               for each1 in os.listdir(path + '/' + each):
                   names.append(each)
                   a_img = Image.open(path + '/' + each + '/' + each1)
                   im = a_img.resize((56, 46), PIL.Image.ANTIALIAS)
                   get_image = im.getdata()
                   column_image = np.reshape(get_image, [-1,1])
                   LDA_train.modify_X(column_image)

    arr_temp = set(names)
    arr = list(arr_temp)
    SW= np.zeros((2576,2576))
    LDA_train.actual_X()
    LDA_train.mean_face_X()
    A = LDA_train.get_X()
    Mean = np.reshape(([0] * 2576), [-1,1])
    
    for i in range(40):
        SWC = np.reshape(([0] * 2576), [-1,1])
        k = i*2
        SWC = np.append(SWC,np.reshape(A[:,k],[-1,1]),axis=1)
        SWC = np.append(SWC,np.reshape(A[:,k+1],[-1,1]),axis=1)
        SWC = np.append(SWC,np.reshape(A[:,k+80],[-1,1]),axis=1)
        SWC = np.append(SWC,np.reshape(A[:,k+81],[-1,1]),axis=1)
        SWC = np.append(SWC,np.reshape(A[:,k+160],[-1,1]),axis=1)
        SWC = np.append(SWC,np.reshape(A[:,k+161],[-1,1]),axis=1)
        SWC = np.append(SWC,np.reshape(A[:,k+240],[-1,1]),axis=1)
        SWC = np.append(SWC,np.reshape(A[:,k+241],[-1,1]),axis=1)
        SWC = SWC[0:,1:]
        

        SMC = [0] * 2576
        for each in range(len(SWC)):
            SMC[each] = float(float(SWC[each,:].sum())/len(SWC[0,:]))
        SMC = array(SMC)
        SMC = np.reshape(SMC,[-1,1])
        Mean = np.append(Mean,SMC,axis=1)
        F_SWC = SWC-SMC
        F_SWC_T = np.transpose(F_SWC)
        prod = np.dot(F_SWC,F_SWC_T)
        SW = SW+prod
   

    print "Shape of SW"SW.shape
    Overall_Mean = [0]*2576
    for each in range(len(A)):
        Overall_Mean[each]= float(float(A[each,:].sum())/len(A[0,:]))
    Overall_Mean = array(Overall_Mean)
    Overall_Mean = np.reshape(Overall_Mean,[-1,1])
    SB = np.zeros((2576,2576))
    for i in range(40):
        mi = np.reshape(Mean[:,i],[-1,1])
        mi_m = mi-Overall_Mean
        mi_m_T = np.transpose(mi_m)
        prdd = np.dot(mi_m,mi_m_T)
        SB = SB+prdd

    SW_I = np.linalg.inv(SW)
    print "Computed Inverse of SW"

    f_prod = np.dot(SW_I,SB)
    print "Calculated SW_I*SB"

    eig_vals, eig_vecs = np.linalg.eig(f_prod)
    print "Shape of the vectors are:"
    print eig_vecs.shape
    index = np.argsort(-eig_vals)
    new_eig_val  = eig_vals[index]
    new_eig_vect = eig_vecs[:,index]
    print new_eig_vect.shape
    print A.shape
##The following two lines help in plotting the graph for eigen values
##    plotter = Plotter('Eigen_values_Sorted')
##    plotter.plot(new_eig_val, 'Eigen Values (Sorted Order)')
    multi_t = np.transpose(new_eig_vect)
    sel_multi_t = multi_t[:20,]
    y = np.dot(sel_multi_t,A)
    print y.shape
    return y,sel_multi_t,names
   

def Testing(test_path,y,p,train_names):
    LDA_test = LDA()
    actual_names = []
    print "Started testing"
    print "Calculating 1NN"
    for path in test_path:
           for each in os.listdir(path):
               for each1 in os.listdir(path + '/' + each):
                   actual_names.append(each)
                   a_img = Image.open(path + '/' + each + '/' + each1)
                   im = a_img.resize((56, 46), PIL.Image.ANTIALIAS)

                   get_image = im.getdata()
                   column_image = np.reshape(get_image, [-1,1])
                   LDA_test.modify_X(column_image)


    LDA_test.actual_X()
    LDA_test.mean_face_X()
    X = LDA_test.get_X()
    print X.shape
    print p.shape
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
        
##        print ('Actual class: ' + actual_names[each] + ' Predicted class: '+ train_names[x])
        if (actual_names[each] == train_names[x]):
            mismatch += 1
        else:
            match += 1
    accuracy = float(match*100/(mismatch + match))
    print 'The accuracy of the Classificaiton: ' + str(accuracy) + ' %'
    print 'Wrong Predictions  : ' + str(match)
    print 'Correct Predictions: ' + str(mismatch)
    return accuracy




start_time = time.time()
print "==========================================================================\n"
print"First Iteration"
y,p,train_names = Train(iteration_1)
acc1 = Testing(test_1, y,p,train_names)
print "==========================================================================\n"
print"First Iteration"
y,p,train_names = Train(iteration_2)
acc2 = Testing(test_2, y,p,train_names)
print "==========================================================================\n"
print"First Iteration"
y,p,train_names = Train(iteration_3)
acc3 = Testing(test_3, y,p,train_names)
print "==========================================================================\n"

print"First Iteration"
y,p,train_names = Train(iteration_4)
acc4 = Testing(test_4, y,p,train_names)
print "==========================================================================\n"

print"First Iteration"
y,p,train_names = Train(iteration_5)
acc5 = Testing(test_5, y,p,train_names)
print "==========================================================================\n"



end_time = time.time()
print 'time taken = ' + str((end_time - start_time)/60) + ' minutes'





















