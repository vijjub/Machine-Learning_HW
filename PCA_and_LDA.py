__author__ ="Vijendra Kumar Bhogadi"
import os, time, sys
from PIL import Image
from PIL import Image as im

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

        self.X                = np.reshape(([0] * 10304), [-1,1])
        self.eigen_val        = None
        self.eigen_vect      = None
        self.new_eigen_val    = None
        self.new_eigen_vect   = None
        self.prod_val         = None
        self.X_mean           = [0] * 10304
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




def Train(iteration):
    pca_train = PCA()
    labels = []
    for path in iteration:
           for each in os.listdir(path):
               for each1 in os.listdir(path + '/' + each):
                   labels.append(each)
                   im = Image.open(path + '/' + each + '/' + each1)
                   get_image = im.getdata()
                   column_image = np.reshape(get_image, [-1,1])
                   pca_train.mod_X(column_image)
    print "starting PCA"
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
    P = pca_train.get_p()
    print "After PCA shape of matrix P "+str(P.shape)
    print "Now starting LDA"
    ##Y is the matrix obtained after PCA
    Y = np.dot(P,pca_train.get_X())
    arr_temp = set(labels)
    arr = list(arr_temp)
    ##calculating Intra class Scaatter matrxi
    print "Calculating SW"
    SW= np.zeros((50,50))
    A = Y
    Mean = np.reshape(([0] * 50), [-1,1])
    print len(A[:,1])
    for i in range(40):
        SWC = np.reshape(([0] * 50), [-1,1])
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
        

        SMC = [0] * 50
        for each in range(len(SWC)):
            SMC[each] = float(float(SWC[each,:].sum())/len(SWC[0,:]))
        SMC = array(SMC)
        SMC = np.reshape(SMC,[-1,1])
        Mean = np.append(Mean,SMC,axis=1)
        F_SWC = SWC-SMC
        F_SWC_T = np.transpose(F_SWC)
        prod = np.dot(F_SWC,F_SWC_T)
        SW = SW+prod
   

    print SW.shape
    Overall_Mean = [0]*50
    for each in range(len(A)):
        Overall_Mean[each]= float(float(A[each,:].sum())/len(A[0,:]))
    Overall_Mean = array(Overall_Mean)
    Overall_Mean = np.reshape(Overall_Mean,[-1,1])
    print "Calulating SB "
    SB = np.zeros((50,50))
    for i in range(40):
        mi = np.reshape(Mean[:,i],[-1,1])
        mi_m = mi-Overall_Mean
        mi_m_T = np.transpose(mi_m)
        prdd = np.dot(mi_m,mi_m_T)
        SB = SB+prdd

    SW_I = np.linalg.inv(SW)
    print "Calculating Inverse of SW"

    f_prod = np.dot(SW_I,SB)
    print "Computed SW_I*SB"

    eig_vals, eig_vecs = np.linalg.eig(f_prod)
    print "Shape of the vectors"
    print eig_vecs.shape
    index = np.argsort(-eig_vals)
    new_eig_val  = eig_vals[index]
    new_eig_vect = eig_vecs[:,index]
    print new_eig_vect.shape
    print A.shape
    multi_t = np.transpose(new_eig_vect)
    p = multi_t[:30,]
    y = np.dot(p,A)
    print "Final dimensions of the training set projected on eigen space "+str(y.shape)
    print y.shape
    return y,p,labels,P
   
    
    



def Testing_KNN(test_path,y,p,train_names,P):
    print "started KNN for One NN"
    pca_test = PCA()
    actual_names = []
    for path in test_path:
           for each in os.listdir(path):
               for each1 in os.listdir(path + '/' + each):
                   actual_names.append(each)
                   im = Image.open(path + '/' + each + '/' + each1)
                   get_image = im.getdata()
                   column_image = np.reshape(get_image, [-1,1])
                   pca_test.mod_X(column_image)

    pca_test.form_X()
    pca_test.subtract_mean_X()
    X = pca_test.get_X()

    ##get mean face of testdata
    print X.shape
    print P.shape


    ##multiplying with P to project on the reduced space
    
    F = np.dot(P,X)
    print F.shape

    ##Multiplying to reduce to LDA
    FF = np.dot(p,F)
    train_data = y
    test_data = FF
    match = 0
    mismatch = 0

    for each in range((test_data.shape[1])):
        lis = []
        for each1 in range(train_data.shape[1]):
            lis.append((la.norm(test_data[:,each] - train_data[:,each1])))

        x = lis.index(min(lis))
        print ('Actual class: ' + actual_names[each] + ' Predicted class: '+ train_names[x])
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
print "==========================================================================\n"
print"First Iteration"
y,p,train_names,P = Train(iteration_1)
acc1 = Testing_KNN(test_1, y,p,train_names,P)
print "==========================================================================\n"

print"Second Iteration"
y,p,train_names,P = Train(iteration_2)
acc2 = Testing_KNN(test_2, y,p,train_names,P)
print "==========================================================================\n"
print"Third Iteration"
y,p,train_names,P = Train(iteration_3)
acc3 = Testing_KNN(test_3, y,p,train_names,P)
print "==========================================================================\n"
print"Fourth Iteration"
y,p,train_names,P = Train(iteration_4)
acc4 = Testing_KNN(test_4, y,p,train_names,P)
print "==========================================================================\n"
print"Fifth Iteration"
y,p,train_names,P = Train(iteration_5)
acc5 = Testing_KNN(test_5, y,p,train_names,P)
print "==========================================================================\n"
total_acc = float((acc1+acc2+acc3+acc4+acc5)/5.0)
print "Final 5-fold accuracy is "+str(total_acc)+"%"
print "==========================================================================\n"
end = time.time()
print 'time taken = ' + str((end - start)/60) + ' minutes'
print "==========================================================================\n"
