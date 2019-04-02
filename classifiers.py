import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from numpy import log,transpose
from numpy.linalg import inv, det

class Discriminant(object):
	
	def __init__(self, path="classification_train.txt"):
		super(Discriminant, self).__init__()
		self.data = pd.read_csv(path,delimiter="\t")
		self.data.columns=["feat1","feat2","label"]
		self.cov = [None,None]
		self.mean = [None,None]
		self.models = [None,None]
	
	def cal_mean_cov(self):
		for i in range(2):
			temp=np.vstack((np.array(self.data[self.data["label"] == i]["feat1"]),
				np.array(self.data[self.data["label"] == i]["feat2"])))
			self.cov[i] = np.cov(temp)
			self.mean[i] = np.mean(temp,axis = 1)

	def visualize(self,area=np.pi*2):
		plt.scatter(self.data[self.data["label"] == 1]["feat1"],self.data[self.data["label"] == 1]["feat2"],s=area)
		plt.scatter(self.data[self.data["label"] == 0]["feat1"],self.data[self.data["label"] == 0]["feat2"],s=area)
		plt.xlabel('feat1')
		plt.ylabel('feat2')
		plt.show()

	def covariance(self):		
		print("covariance of class 0: ",self.cov[0],"\ncovariance of class 1: ",self.cov[1])

	def meanp(self):
		print("mean of class 0: ",self.mean[0],"\nmean of class 1: ",self.mean[1])

	def model(self,x):
		for i in range(2):
			prob = len(self.data[self.data["label"]==i]) /len(self.data)
			detCov = det(self.cov[i])
			X = x-self.mean[i]
			linear = np.matmul(np.matmul(transpose(X),inv(self.cov[i])),
				X)
			g = log(prob)-(1/2)*log(detCov)
			g -= 1/2* linear
			yield g

	def test_model(self,path="classification_test.txt"):
		test_data = pd.read_csv(path,delimiter="\t")
		test_data.columns=["feat1","feat2","label","_"]	#because there is a un expected "\t" in classification_test.txt
		accurate_pred = 0
		len_test = len(test_data)
		for i in range(len_test):
			label = test_data["label"][i]
			x = np.array((test_data["feat1"][i],
				test_data["feat2"][i]))
			c0,c1 = self.model(x)
			if(c0>=c1 and 0 == label):
				accurate_pred += 1
			elif(c0<c1 and 1 == label):
				accurate_pred += 1
		return accurate_pred/len_test*100

class Regression(object):
	data = {"head":[],"brain":[]}
	def __init__(self, path="regression_data.txt.",initial=[1.,1.]):
		super(Regression, self).__init__()
		
		with open(path,"r") as file:
			file.readline() #to get rid of first line
			for line in file:
				data = line.split()
				try:
					self.data["head"].append(float(data[0])/1000)
					self.data["brain"].append(float(data[1])/1000)
				except Exception:
					pass
			self.data = pd.DataFrame(self.data)	#coping all data to ram and creating dataframe is done
		self.w = np.array(initial) #initial values
		self.folds = []

	def creating_folds(self,n_fold=5):
		shuffled = self.data.iloc[np.random.permutation(len(self.data))]
		shuffled.reset_index(drop=True,inplace=True)
		counter = 0
		length=len(self.data)
		for i in range(n_fold):
			temp = []
			while ((i/5)*length<= counter < ((i+1)/5)*length):
				temp.append([[1.,self.data["head"][counter]],self.data["brain"][counter]])
				counter += 1
			self.folds.append(list(temp))

	def evaluate_model(self):
		total_mse=0.
		print("------------------------------------------------")
		for i in range(len(self.folds)):
			stack = []
			xxw = 0.0
			xy = [0.,0.]
			for j in range(len(self.folds)):
				if(i != j):
					for k in range(len(self.folds[j])):
						xxw += 2 * np.matmul(self.folds[j][k][0],transpose(self.folds[j][k][0])) *self.w
						xy += 2 * transpose(self.folds[j][k][0]) * self.folds[j][k][1]
			#first repeat optimization 15 times for step 10^-5
			for repeat in range(15):
				self.optimizer(xxw,xy)
			#then repat optimization 50 times for step 10^-6
			for repeat in range(50):
				self.optimizer(xxw,xy,step=0.000001)
			#in end repeat optimization 100 for step 10^-7
			for repeat in range(100):
				self.optimizer(xxw,xy,step=0.0000001)
			#decreasing step size help model to converge minimum point
			#sorry for coding this part messy. but i don't have time(i am a bit lazy:)) 
			mse = self.mse()
			total_mse += mse  
			print("\n",i+1,".cross Validation \nmse: ",mse)

		print("\n\nOverall: ",total_mse/len(self.folds))
		print("------------------------------------------------")

					
	def optimizer(self,xx,xy,step = 0.00001):
		gradient = xx - xy 
		self.w = self.w - step * gradient #there is no problem when xx is invertible

	def mse(self):
		result = 0.0
		for i in range(len(self.data)):
			result += self.data["head"][i] * self.w[1] + self.w[0] - self.data["brain"][i]
		result = result**2
		result /= len(self.data)
		return result

if __name__ == '__main__':
	print("to see result please run main.py ;)\n...")