Documentation for the Homework 3:
--------------------------------------------
Name : Vijendra Kumar Bhogadi
UTA-ID: 1001052460

There are seven python scripts attached 
1) partition.py -- Running this code will distribute the data into 5 sets of equal size
2) Knn.py -- Implemented using PCA and 1_KNN. Avg 5-fold accuaracy is 97%
3) KNN_resize.py -- Implemented using PCA and 1-KNN but with images of size 56*46. Overall accuracy is 96%
4) LDA_pure.py -- Implemente LDA and 1-NN but with resized images. The accuracy falters from 56% to 76%
5) PCA_and_LDA -- Implemetd PCA and then LDA with 1-NN again. Accuracy is 98% on avg over all iterations
6) SVM_face.py -- Normal SVM using gaussian kernel implemented with 5-fold cross validation. Accuracy is 95% avg
7) SVM_PCA.py -- Implemneted PCA and then SVM. The accuracy is 98 to 99% on avg on all runs.

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
4) Now before feeding the feature vector to trainer we do a dimensionality reduction by using PCA and LDA and also running the algorithms using 5-fold cross validation method.
5) in both LDA and PCA find the eigen vectors correspponding to their importance.
for LDA -- eigen-vectors = 20
for PCA -- eigen-vectors = 50

Conclusion:
----------------
From the above tasks it is clear that SVM is more sensitive to high dimendionality of the data.

References:
----------
1) Andrew Tulloch-- Soft margin SVM Tutorial("http://tullo.ch/articles/svm-py/".)
2) Machine Learning -- Tom Mitchell
3) Python documentation--https://docs.python.org/2/
4) Scikit learn documemtation ---http://scikit-learn.org/stable/index.html
5) http://www.cs.columbia.edu/~kathy/cs4701/documents/jason_svm_tutorial.pdf
6) http://sebastianraschka.com/Articles/2014_pca_step_by_step.html
7)http://sebastianraschka.com/Articles/2014_python_lda.html
