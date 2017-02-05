__author__ ="Vijendra Kumar Bhogadi"
import os,shutil,math

path = "/home/vijjub/Documents/Machine Learning/Homework 2"
print("Starting data partition into test and train")
##Creating new directories for test and train

train_path = path+"/train"
if not os.path.exists(train_path):
    os.makedirs(train_path)

test_path = path+"/test"
if not os.path.exists(test_path):
    os.makedirs(test_path)

##Paritioning of Training and testing data into 50:50 partition

orl_directory = "/home/vijjub/Documents/Machine Learning/Homework 2/orl_faces"

for dirs in os.listdir(orl_directory):
    group = orl_directory+"/"+dirs
    os.makedirs(train_path+"/"+dirs)
    os.makedirs(test_path+"/"+dirs)
    count = math.ceil(0.5*len(os.listdir(group)))
    c =0
    for files in os.listdir(group):
        srcfile = group+"/"+files
        if(c<count):
            shutil.copy(srcfile,train_path+"/"+dirs)
            c = c+1
        else:
            shutil.copy(srcfile,test_path+"/"+dirs)
        
print("Completed paritionaing into testing and training data")

