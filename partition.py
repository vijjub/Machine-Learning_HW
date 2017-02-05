import os,shutil,math

path = "/home/vijjub/Documents/Machine Learning/Homework 2"
print("Starting data partition into test and train")
##Creating new directories for test and train

part_1 = path+"/t1"
if not os.path.exists(part_1):
    os.makedirs(part_1)

part_2 = path+"/t2"
if not os.path.exists(part_2):
    os.makedirs(part_2)

part_3 = path+"/t3"
if not os.path.exists(part_3):
    os.makedirs(part_3)

part_4 = path+"/t4"
if not os.path.exists(part_4):
    os.makedirs(part_4)


part_5 = path+"/t5"
if not os.path.exists(part_5):
    os.makedirs(part_5)


##Paritioning of Training and testing data into 50:50 partition

orl_directory = "/home/vijjub/Documents/Machine Learning/Homework 2/orl_faces"

for dirs in os.listdir(orl_directory):
    group = orl_directory+"/"+dirs
    os.makedirs(part_1+"/"+dirs)
    os.makedirs(part_2+"/"+dirs)
    os.makedirs(part_3+"/"+dirs)
    os.makedirs(part_4+"/"+dirs)
    os.makedirs(part_5+"/"+dirs)
    
    
    c =0
    for files in os.listdir(group):
        srcfile = group+"/"+files
        if(c<2):
            shutil.copy(srcfile,part_1+"/"+dirs)
            c = c+1
        elif (c>=2 and c<4):
            shutil.copy(srcfile,part_2+"/"+dirs)
            c = c+1
        elif (c>=4 and c<6):
            shutil.copy(srcfile,part_3+"/"+dirs)
            c = c+1
        elif (c>=6 and c<8):
            shutil.copy(srcfile,part_4+"/"+dirs)
            c = c+1
        else:
            shutil.copy(srcfile,part_5+"/"+dirs)
        
print("Completed paritionaing into testing and training data")

