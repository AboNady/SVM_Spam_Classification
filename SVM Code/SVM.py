#import the needed libraries
import time
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import pandas as pd

start = time.time()

#Why I used index_col? to delete the ID's Coulmn that Panadas create automatically..  try without it and see what will happen!
dataa = pd.read_csv(r'C:\Users\NADY\Desktop\archive\emails.csv',index_col = 0 ) # Put your path here

inp =   dataa.iloc[:,0:-1] # My input data [ ALL Rows × Columns(exept yhe last one as it's the output one) ]
oup =   dataa.iloc[:,-1]   # Output [ ALL Rows × The last Column ONLY(it's the output) ] 1 for spam, 1 for not spam
tst_size = 0.30            # The Percentage of the Test size *note that the train size is 1-tst_size

# Separate the data to train and test
X_train, X_test , y_train, y_test = train_test_split(inp, oup, test_size = tst_size)

# Initialize the model
clf = svm.SVC(kernel='linear',  C = 10)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)


# Some usefel information about the model
acc  = metrics.accuracy_score(y_test, y_pred)
recl = metrics.recall_score(y_test , y_pred)
scr  = clf.score(X_test, y_test)



#Printing the results
print(  'Test Size is:  ' , tst_size )
print(  'Accuracy is:  ' , acc )
print(  'Recall is:  ' , recl )
print(  'Score is:  ' , scr )

# To calculate the Execution time
end = time.time()
exctim = end-start
print('Excuation time is =>', exctim , 'Seconds<=')


 
