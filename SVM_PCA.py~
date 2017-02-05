__author__ ="Vijendra Kumar Bhogadi"
import os, time, sys
from PIL import Image as im
from PIL import Image

# Linear Algebra Library
from numpy import *
from numpy.linalg import *
import numpy as np
import numpy.linalg as la
#Solver
import math
import cvxopt
import cvxopt.solvers

##Folder locations of the dataset

path_part_1 = "/home/vijjub/Documents/Machine Learning/Homework 2/t1"
path_part_2 = "/home/vijjub/Documents/Machine Learning/Homework 2/t2"
path_part_3 = "/home/vijjub/Documents/Machine Learning/Homework 2/t3"
path_part_4 = "/home/vijjub/Documents/Machine Learning/Homework 2/t4"
path_part_5 = "/home/vijjub/Documents/Machine Learning/Homework 2/t5"


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




def get_classes(each, length):
    k=each*2
    classes = np.ones((1,length))
    classes = classes*-1
    classes[0,(k)]=1
    classes[0,(k+1)]=1
    classes[0,(k+80)]=1
    classes[0,(k+81)]=1
    classes[0,(k+160)]=1
    classes[0,(k+161)]=1
    classes[0,(k+240)]=1
    classes[0,(k+241)]=1
    return classes



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



    def modify_X(self, vect):
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





##Training by using one vs All classification method
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
                   pca_train.modify_X(column_image)

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
    machines = {}
    for i in range(40):
        classes = get_classes(i, len(labels))
        final = {}
        
        sol,sup_init, sup_vect, sup_init_y, index, mul, b = get_train_results(y.T,classes)
        final['sol'] = sol
        final['sup_init'] = sup_init
        final['sup_vect'] = sup_vect
        final['sup_init_y'] = sup_init_y
        final['index'] = index
        final['mul'] = mul
        final['b'] = b
        machines[labels[i*2]] = final
        print '\nCompleted Training for class : ' + str(labels[i*2])
        print 'No.of support vectors for this data : ' + str(len(sup_vect))
        print 'Total no.of trained classes: '+str(i+1)
        print 'Please wait........... Training next class......'

    return machines,p






def Testing(test_path,machines,p):
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
                   pca_test.modify_X(column_image)

    pca_test.form_X()
    pca_test.subtract_mean_X()
    X = pca_test.get_X()

    ##get mean face of testdata
    print X.shape
    print p.shape


    ##multiplying with p to project on the reduced space
    
    F = np.dot(p,X)
    print F.shape

    feature_vector = F.T
    summ=0
    match = 0
    mismatch = 0

    for i in feature_vector:
        results = []
        peak_val = -999999
        for key in machines:
            value = new_image_classify(i, machines[key]['sup_init'], machines[key]['sup_init_y'], machines[key]['sup_vect'],machines[key]['index'], machines[key]['b'])
            if(value > peak_val ):
                peak_val = value
                new_key = key

       
        print 'Actual Class: ' + str(actual_names[summ]) +'\t'+ ' Predicted Class: ' + str(new_key)
        if(actual_names[summ] == new_key):
            match = match + 1
        else:
            mismatch = mismatch+1
        summ = summ + 1

    acc = (float(match)/80.0)*100
    print 'Total Matches are: ' + str(match)
    print 'Total Wrong Matches are : ' + str(mismatch)
    print 'Accuracy : ' + str(acc)+'%'
    return acc
    

    
       
#Using 5-fold cross validation
print "====================================================================================\n\n"
print "Iteration 1"
print "Started Training the Classifier"
machines1,p1 = Train(iteration_1)
print "Started testing classifier with other 1 part data"
pctg1 = Testing(test_1, machines1,p1)

print "====================================================================================\n"
print "Iteration 2"
print "Started Training the Classifier"
machines2,p2 = Train(iteration_2)
print "Started testing classifier with other 1 part data"
pctg2 = Testing(test_2, machines2,p2)

print "====================================================================================\n"
print "Iteration 3"
print "Started Training the Classifier"
machines3,p3 = Train(iteration_3)
print "Started testing classifier with other 1 part data"
pctg3 = Testing(test_3, machines3,p3)

print "====================================================================================\n"
print "Iteration 1"
print "Started Training the Classifier"
machines4,p4 = Train(iteration_4)
print "Started testing classifier with other 1 part data"
pctg4 = Testing(test_4, machines4,p4)

print "====================================================================================\n"
print "Iteration 5"
print "Started Training the Classifier"
machines5,p5 = Train(iteration_5)
print "Started testing classifier with other 1 part data"
pctg5 = Testing(test_5, machines5,p5)

print "====================================================================================\n"



##Final avg accuracy of the classificationprint "calculating avg accuaracy of all 5 " 
final_avg = (float(pctg1+pctg2+pctg3+pctg4+pctg5)/5.0)
print "Total Avg percentage accuracy is :"+str(final_avg)+'%'
