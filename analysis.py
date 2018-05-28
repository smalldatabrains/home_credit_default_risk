import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier


def main():
	print("HOME CREDIT DEFAULT RISK")
	#listing available files
	cwd=os.getcwd()
	cwd=cwd+"/data"
	os.chdir(cwd)
	files=os.listdir(cwd)
	print(files)
	description=csv_read('HomeCredit_columns_description.csv')
	

	print("ANALYSIS")
	for file in files:
		df=csv_read(file)
		df.name=file[:-4]
	#creates a figure for each numerical variable
		for variable in df.columns:
			if(df[variable].dtype==np.float64 or df[variable].dtype==np.int64):
				print(variable)
				chart_numerical(df,variable,df.name)
			

			elif(df[variable].dtype==object):
				print(variable," is a categorical value")
				chart_categorical(df,variable,df.name)

	#generate statistical values
		statistics=stats(df,df.name)

	#generate correlation map
		correlation(df,df.name)

	#joining tables

	#features selection

	print("TRAINING THE MODEL")
	#prepare data for training

	#train different models and compare performance


#csv reader and informations
def csv_read(file):
	data=pd.read_csv(file)
	#before removing NaN
	print("Before removing missing values",data.shape)
	#print information about NaN and Null values
	variable_status(data)
	data.dropna() #removing NaN values for first analysis
	print("After removing missing values",data.shape)
	print(data.columns)
	return data

#check qty/ratio of NaN and Null values for a variable
def variable_status(data):
	count=data.isnull().sum()
	total=data.shape[0]
	ratio=count/total
	return pd.DataFrame(data={'missing value':count,'ratio':ratio},index=data.columns.values)

#data analysis on dtype=float64 and int64
def chart_numerical(df,x,name):
	fig=plt.figure()
	df[x].hist(bins=50)
	plt.title(x)
	if not os.path.exists('../fig/'+name+'/'):
		os.makedirs('../fig/'+name+'/')
	fig.savefig('../fig/'+name+'/'+x+'.png')
	plt.close(fig) #for memory management purpose

def chart_categorical(df,x,name):
	fig=plt.figure()
	counts=df[x].value_counts()
	counts.plot(kind='pie')
	plt.axis('equal')
	plt.title(x)
	if not os.path.exists('../fig/'+name+'/'):
		os.makedirs('../fig/'+name+'/')
	fig.savefig('../fig/'+name+'/'+x+'.png')
	plt.close(fig) #for memory management purpose

def stats(df,name):
	stats=pd.DataFrame(columns=['variable','mean','median','std','range','skewness','kurtosis'])
	for x in df.columns:
		if(df[x].dtype==np.float64 or df[x].dtype==np.int64):
		#calculate estimators on variables and store it into an array
			mean=np.mean(df[x])
			median=np.median(df[x])
			std=np.std(df[x])
			rg=np.ptp(df[x])
			skewness=df[x].skew()
			kurtosis=df[x].kurtosis()
			stats.loc[-1]=[x,mean,median,std,rg,skewness,kurtosis]
			stats.index=stats.index+1
	stats.to_csv('../statistics/statistics_'+name+'.csv')
	return stats


#feature selection for training
#create csv_file with selected inputs and labels per loan (merging of informations)
def features_selection(df):
	pass


def scatter_matrix(df):
	pass

def correlation(df,name):
	correlation=df.corr()
	fig=plt.figure()
	sns.heatmap(correlation.values)
	plt.show()
	plt.title('correlation '+name)
	fig.savefig('../correlation/'+name+'.png')
	plt.close(fig)


#logistics regression
def logistic_regression(inputs,labels,X_test,y_test):
	classifier=linear_model.logisticRegression()
	classifier.fit(inputs,labels)
	accuracy=classifier.score(X_test,y_test)
	print(accuracy)

#random forest
def random_forest(inputs,labels,X_test,y_test):
	classifier=RandomForestClassifier(n_estimators=10)
	classifier.fit(inputs,labels)
	accuracy=classifier.score(X_test,y_test)
	print(accuracy)

#keras model
def rnn(inputs,labels):
	pass

if __name__=="__main__":
	main()
else:
	print("Module imported")