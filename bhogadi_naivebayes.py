__author__ ="Vijendra Kumar Bhogadi"


import os,shutil,math
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from collections import Counter
import re

docs = []
##For running the code please use directory locations from your machine
print("Building a Vocabulary")

## building a vocabulary from the training data using sklearn.Countvectorizer
for dirs in os.listdir("/home/vijjub/Documents/Machine Learning/homework 1/train"):
	dire = "/home/vijjub/Documents/Machine Learning/homework 1/train/"+dirs
	for files in os.listdir(dire):
		filename = dire+"/"+files
		eachfile = open(filename)
		docs.append(eachfile.read())
		
c_vect = CountVectorizer(decode_error=u'ignore',stop_words='english',min_df=1)
features = c_vect.fit_transform(docs)
all_tokens = c_vect.get_feature_names()
stemmed_tokens = []
stemmer = PorterStemmer()

##Stemming all the tokens for improving the accuracy
for i in range(len(all_tokens)):
        stemmed_tokens.append(stemmer.stem(all_tokens[i]))



final_voc = list(set(stemmed_tokens))
size_of_voc = len(final_voc) 

print("Size of the Vocabulary: "+str(size_of_voc))

classes = os.listdir("/home/vijjub/Documents/Machine Learning/homework 1/train")
matrix_dict = {}
size_of_bag = {}

cachedStop = set(stopwords.words("english"))
direc = "/home/vijjub/Documents/Machine Learning/homework 1/train/"

print("Removed Stop words and Also stemmed all the tokens")
print("Please wait for a few secons in process")

##Calculating the count of each word in each class
for val in classes:
        folder = direc + val + "/"
        tokens = []
        word_dict = {}
        for word in final_voc:
                word_dict[word]=0
        
        for files in os.listdir(folder):
                filename = folder+files
                eachfile = open(filename)
                doc = re.split('\W+',eachfile.read())
                tokens.extend(doc)


        flist = []
        for i in range(len(tokens)):
                try:
                        word = stemmer.stem(tokens[i])
                        flist.append(word)
                except UnicodeDecodeError:
                        continue
        ##print("started stop word removal")
        fflist = []
        fflist = [word for word in flist if word not in cachedStop]
        size_of_bag[val] = len(fflist)
        count_dict = dict(Counter(fflist))
        ##print("started frequency counter")
        for key in word_dict:
                try:
                        count = count_dict[key]
                        word_dict[key] = count
                except KeyError:
                        continue

        matrix_dict[val] = word_dict

                

## Calculating Prior probabilities P(c)
print("Calculating Prior probabilities of each class")
prior = {}
for val in classes:
        prob = float((500.0)/(10000.0))
        prior[val] = prob



##Calculation of Likelihood estimate of each word in that class
cond_prob = {}
temp_dict = {}

for each in classes:
        temp_dict = {}
        for val in final_voc:
                con_prob = (matrix_dict[each][val]+1.0)/float(size_of_voc+size_of_bag[each])
                temp_dict[val]= con_prob

        cond_prob[each] = temp_dict

##Prediction preparations
test_dict = {}
test_directory = "/home/vijjub/Documents/Machine Learning/homework 1/test"
for fol in os.listdir(test_directory):
        folder = test_directory +"/"+fol
        for files in os.listdir(folder):
                filename = folder+"/"+files
                eachfile = open(filename)
                doc = eachfile.read()
                test_dict[files] = [doc,fol]
                
print("Started Classification of test documents")
print("Calculating Likelihood estimate of each of the test document")


##Calculation of the probabilities of each document with respect to each class and taking the max probability
for key in test_dict:
        words = re.split('\W+',test_dict[key][0])
        words_nostop = [word for word in words if word not in cachedStop]
        words_flist = []
        for i in range(len(words_nostop)):
                try:
                        word = stemmer.stem(words_nostop[i])
                        words_flist.append(word)
                except UnicodeDecodeError:
                        continue

        temp_d = dict(Counter(words_flist))
        f_dic = {}
        
        for cat in classes:
                p = 1.0
                for word in temp_d:
                        try:
                                p = p + temp_d[word]*math.log(float(cond_prob[cat][word]))
                        except KeyError:
                                continue
                f_prob = p* prior[cat]
                f_dic[cat] = f_prob

        f_class = max(f_dic,key= f_dic.get)
        test_dict[key].append(f_class)


##Printing the results to a text document

print("Printing the results to a text file")
f = open('/home/vijjub/Documents/text.txt','w')
f.write("Document_name \t Actual Class \t Predicted Class")
f.write("----------------------------------------------------------")
for key in test_dict:
	f.write(key+'\t'+test_dict[key][1]+'\t'+test_dict[key][2]+'\n')

##Calculating the accuracy 
print("Calculating Accuracy of the classification" )
count = 0
for key in test_dict:
        if(test_dict[key][1]==test_dict[key][2]):
                count = count+1
        else:
                continue

accuracy = float(count/float(len(test_dict.keys())))*100
print("Accuracy is : "+str(accuracy))


        
                
                
                
        
        
        

                
                
        
                
                
        



        
        

