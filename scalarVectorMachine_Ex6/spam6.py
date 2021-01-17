####################################################################################
"""     Machine Learning Online Class Exercise 6 | Support Vector Machines      """
"""     Machine Learning Online Class Exercise 6 | Support Vector Machines      """
####################################################################################
######### =============== Part 2: spam classification ================
#                    use SVMs to build your own spam filter.
# import modules
import scipy.io as io
import scipy as sp
from  scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm       #SVM software
import re #regular expression for e-mail processing
# import nltk
from stemming.porter2 import stem

####################################################################################
##################  essentialFunctions  ##################
# get the vocabolary list           ########################
def getVocabdict(reverse=False):
    # index_vocab_list= np.loadtxt('./data/vocab.txt', dtype=str)
    # # we have to change the array to the index_vocab_dict
    # index_vocab_dict ={(row[1]):int(row[0]) for row in index_vocab_list}
    # return index_vocab_dict
    vocab_dict = {}
    with open("./data/vocab.txt") as f:
        for line in f:
            (val, key) = line.split()
            if not reverse:
                vocab_dict[key] = int(val)
            else:
                vocab_dict[int(val)] = key

        return vocab_dict


# regular Expression           ########################
emailRegex = re.compile(r'''(
    [a-zA-Z0-9._%+-]+               # username
    @                               # @  symbol
    [a-zA-Z0-9.-]+                  # domain name
    (\.[a-zA-Z]{2,4})               #dot - something
    )''', re.VERBOSE)
urlRegex = re.compile(r'''(
    (http[s]?://www\.)+              #http
    [a-zA-Z0-9.-]+                  # domain name
    (\.[a-zA-Z]{2,4}[/]?)               #dot - something
    )''', re.VERBOSE | re.IGNORECASE )
numRegex = re.compile(r'''([0-9]+)''' )
dollarRegex = re.compile(r'([$]+)')
htmlRegex = re.compile(r'(<[/a-z]*>)',re.VERBOSE)

# clean the email       ########################
def processEmail(emailSample):
    e= emailSample.lower()
    e = emailRegex.sub('emailaddr', e)
    e = urlRegex.sub('httpaddr', e)
    e = numRegex.sub('number', e)
    e = dollarRegex.sub('dollar', e)
    # dolarnumber can be classified as money which is great in my thought
    e = htmlRegex.sub('', e)
    # Handle punctuation and special ascii characters
    e = re.sub(r'[@$/\\#,-:&*+=\[\]?!(){}\'\">_<;%]+', '',e)
    # Handle URLS             # email_contents = re.sub(r'(http|https)://[^\s]*', 'httpaddr', email_contents)
    # Handle Email Addresses  # email_contents = re.sub(r'[^\s]+@[^\s]+', 'emailaddr', email_contents)
    return e

# tokenize the email        ########################
def tokenizeEmail(emailSample):
    emailstk=[]
    tokens =re.split('[ \n]', emailSample)
    for token in tokens:
        if not len(token): continue
        try:
            token = stem(token)
        except:
            continue
        emailstk.append(token)
    return emailstk

# indexes the email word -tokens   ########################
def email2vocabolaryIndecs(emailSample, index_vocab_dict):
    tokenlist = tokenizeEmail(processEmail(emailSample))
    indexList = [index_vocab_dict[token] for token in tokenlist if token in index_vocab_dict ]
    print(len(indexList))
    return indexList

# Extracting features from email      ########################
def emailFeature(emailIndex, voc):
    x = np.zeros(len(voc))
    for ind in emailIndex:
        x[ind] = 1
    return x
####################################################################################
####################################################################################
###########             testing the above functions....
def testFunctions():
    index_vocab_dict = getVocabdict()
    emailSample=''
    # get sample1
    with open('./data/emailSample1.txt', 'r') as emailSam:
        # emailSam.write(' jack@gmail.com    httPs://www.jacky.com  9807890  <html> hello world </html>')
        emailSample = emailSam.read()
        emailSam.close()
    print(emailSample)
    emailIndex = email2vocabolaryIndecs(emailSample, index_vocab_dict)
    x_test = emailFeature(emailIndex, index_vocab_dict)
    # 2 this  value is not counting i dont know why?
    # diff ====>  thi(exercise-token)  != this  (my-token)
    print ("Length of feature vector is %d" % len(x_test))
    print ("Number of non-zero entries is: %d" % sum(x_test==1))

###########             training svm for spam classification
def trainEamil():
    print('trianing svm ...............')
    # Training set
    mat = io.loadmat('./data/spamTrain.mat')
    X, y = mat['X'], mat['y']
    #NOT inserting a column of 1's in case SVM software does it for me automatically...
    #X =     np.insert(X    ,0,1,axis=1)

    # Test set
    mat = io.loadmat('./data/spamTest.mat')
    Xtest, ytest = mat['Xtest'], mat['ytest']

    pos = np.array([X[i] for i in range(X.shape[0]) if y[i] == 1])
    neg = np.array([X[i] for i in range(X.shape[0]) if y[i] == 0])

    print('Total number of training emails = ',X.shape[0])
    print('Number of training spam emails = ',pos.shape[0])
    print('Number of training nonspam emails = ',neg.shape[0])
    # Total number of training emails =  4000
    # Number of training spam emails =  1277
    # Number of training nonspam emails =  2723

    # Run the SVM training (with C = 0.1) using SVM software.
    # First we make an instance of an SVM with C=0.1 and 'linear' kernel
    linear_svm = svm.SVC(C=0.1, kernel='linear')
    # Now we fit the SVM to our X matrix, given the labels y
    linear_svm.fit( X, y.flatten() )
    # "Once the training completes, you should see that the classifier gets a
    #  training accuracy of about 99.8% and a test accuracy of about 98.5%"

    train_predictions = linear_svm.predict(X).reshape((y.shape[0],1))
    train_acc = 100. * float(sum(train_predictions == y))/y.shape[0]
    print('Training accuracy = %0.2f%%' % train_acc)

    test_predictions = linear_svm.predict(Xtest).reshape((ytest.shape[0],1))
    test_acc = 100. * float(sum(test_predictions == ytest))/ytest.shape[0]
    print('Test set accuracy = %0.2f%%' % test_acc)
    return  linear_svm, pos, neg

###########      Top Predictors for Spam
def topPrediction():
    linear_svm, pos, neg = trainEamil()
    vocab_dict_flipped = getVocabdict(reverse=True)
    #Sort indicies from most important to least-important (high to low weight)
    sorted_indices = np.argsort(linear_svm.coef_, axis=None )[::-1]
    print ("The 15 most important words to classify a spam e-mail are:")
    print( [ vocab_dict_flipped[x] for x in sorted_indices[:15] ])
    print ("The 15 least important words to classify a spam e-mail are:")
    print ([ vocab_dict_flipped[x] for x in sorted_indices[-15:] ])


    # Most common word (mostly to debug):
    most_common_word = vocab_dict_flipped[sorted_indices[0]]
    print( '# of spam containing \"%s\" = %d/%d = %0.2f%%'% \
        (most_common_word, sum(pos[:,1190]),pos.shape[0],  \
         100.*float(sum(pos[:,1190]))/pos.shape[0]))
    print ('# of NON spam containing \"%s\" = %d/%d = %0.2f%%'% \
        (most_common_word, sum(neg[:,1190]),neg.shape[0],      \
         100.*float(sum(neg[:,1190]))/neg.shape[0]))

####################################################################################
####################################################################################
############################    happy hacking :)    ################################
####################################################################################
####################################################################################
####################################################################################
