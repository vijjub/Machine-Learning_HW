Documentation for the Homework 1:
--------------------------------------------
Name : Vijendra Kumar Bhogadi
UTA-ID: 1001052460

There are two python scripts attached 
1) bhogadi_partition.py -- Running this code will distribute the data into test and training with 50:50 split
2) bhogadi_SVM_face.py -- Running this code starts the actual implementation of the algorithm

Execution Details:
------------------

To run the code just open with 'Python Idle' which is a builtin python ide and run.
All the directory names or filepaths in the scripts must be replaced with the directory names or paths on your local machine otherwise an error will be raised

Implementation Details:
-----------------------

1) Constructed a feature vector of each image using PIL.Image
2) Implemented One vs All training for the classifier.So created labels for the images like +1 for the image belonging to the same class and -1 if  
   belonging to different class
3) Now we feed our feature vector matrix of all the images and their respective labels to trainer and solve the Lagrangian using cvxopt library and 
   calculate the solution.
4) Now we find the support vectors using the above solution.
5) We also calculate bias using the support vectors .
6) Once the training is completed for all the 40 classes we start testing it with the test faces and Calculate accuracy.
7) Now swap the train and test data and train the classifier using the test data.
8) After the training is complete test with training data and calculate accuracy. Take the average of both the accuracy and get the final accuracy 
   value
9) Accurracy varies between 87% to 95%.


References:
----------
1) Andrew Tulloch-- Soft margin SVM Tutorial("http://tullo.ch/articles/svm-py/".)
2) Machine Learning -- Tom Mitchell
3) Python documentation--https://docs.python.org/2/
4) Scikit learn documemtation ---http://scikit-learn.org/stable/index.html
5) http://www.cs.columbia.edu/~kathy/cs4701/documents/jason_svm_tutorial.pdf
