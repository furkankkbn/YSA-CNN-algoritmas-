import json
import cv2
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D

class CNNmodel():

	'''
		başlatmak için gerekli tanımlamalar
	'''
	def __init__(self,setpath,allpath='data'):
		self.allpath=allpath		
		self.setpath=setpath		
		self.connpath=None			
		self.kinds=[]				
		self.patharray=[]			
		self.picarray=[]			
		self.piccount=[]			
		self.labels=[]				
		self.dictionary=[]			

	'''
		
        Resim toplama klasörünü döndür
	'''
	def baglanti(self):
		
		self.connpath = os.path.join(self.allpath,self.setpath)
		print('connpath:',self.connpath)
		
		filelist = os.listdir(self.connpath)
		for file in filelist:
			print('file:',file)
			filedir = os.path.join(self.connpath,file)
			
			if(os.path.isdir(filedir)):
				self.kinds.append(file)
				self.patharray.append(filedir)
	'''
		Veri okuma

	'''
	def veriOku(self):
		count = 0

		for path in self.patharray:
			piclist = os.listdir(path)
			self.piccount.append(len(piclist))
			print('piccount:',self.piccount)
			for pic in piclist:
				picdir = os.path.join(path,pic)
				print('picdir:',picdir)
				picarray= cv2.imread(picdir,cv2.IMREAD_GRAYSCALE)
				picarray = cv2.resize(picarray,(32,32),interpolation=cv2.INTER_CUBIC)
				#picarray = picarray.flatten()
				print('picarray:',np.array(picarray).shape)
				self.picarray.append(picarray)
			print('selfpic:',np.array(self.picarray).shape)

		for i in self.piccount:
			for j in range(i):
				self.labels.append(count)
			count+=1
		print('labels',self.labels)

	'''
		dizi türünü değiştirme
	'''
	def tipdegistir(self,types='nparray'):
		if types=='nparray':
			nparray = np.array(self.picarray)
			return nparray

	'''
		normazlize etme
	'''
	def normalize(self,types='MINMAX'):
		if types=='MINMAX':
			big = 0
			small = 255
			length = len(self.picarray)
			print('length:',length)
			array = []
			for i in range(length):
				array.append([])

			for num in self.picarray:
				for row in num:
					for col in row:
						if col>big:
							big = col
						if col<small:
							small = col
			print('big:',big,',small:',small)
			count = 0
			rowcount = 0
			colcount = 0
			for num in self.picarray:
				for row in num:
					for col in row:
						self.picarray[count][rowcount][colcount]=(col-small)/(big-small)
						colcount+=1
					colcount=0
					rowcount+=1
				rowcount=0
				count+=1	
			#print('arraylen:',np.array(array).shape)
			#self.picarray = array
			print('picarraynew:',np.array(self.picarray).shape)


	'''
		verileri encord etme
	'''
	def one_hot(self):
		x = np.array(self.labels)
		#print('eye:',np.eye(len(self.piccount))[x])
		self.labels = np.eye(len(self.piccount))[x]
		print('label_eye:',self.labels)



	def sozlukayarla(self):
		count = 0
		index = 0
		#print(len(self.dictionary))	
		for i in self.piccount:
			self.dictionary.append([])
			for number in range(i):
				self.dictionary[count].append({"filename":self.kinds[count],"datas":self.picarray[index].tolist(),"labels":self.labels[index]})
				index+=1
			count+=1

	def jsonpaket(self):
		for i in range(len(self.piccount)):
			print('jsonpack:',len(self.dictionary[i]))
			with open(self.connpath+"\\datas_%s.json"%self.kinds[i],'w') as file:
				json.dump(self.dictionary[i],file)
	'''
		json dosya bilgileri yüklenir
	'''
	def jsonyukle(self):
		for file in os.listdir(self.connpath):
			if os.path.isfile(os.path.join(self.connpath,file)):
				with open(os.path.join(self.connpath,file),'r') as f:  
					batch =json.load(f)
					#print('batch:',batch[1]['filename'])

	def modelolustur(self,model):
		model.add(Conv2D(filters=2,kernel_size=(3,3),padding='same',input_shape=(32,32,1),activation='relu'))
		model.add(Dropout(0.25))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Conv2D(filters=2,kernel_size=(3,3),padding='same',activation='relu'))
		model.add(Dropout(0.25))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Flatten())
		model.add(Dense(3,activation='relu'))
		model.add(Dropout(0.25))
		model.add(Dense(len(self.kinds),activation='softmax'))
		return model  

	def ozetal(self,model):
		return model.summary()

	def train(self,model,Xtrain,Ytrain):
		model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
		model.fit(x=Xtrain,y=Ytrain,validation_split=0.0,epochs=10,batch_size=2,verbose=2)
		return model

	def degerlendirme(self,model,Xtrain,Ytrain):
		return model.evaluate(Xtrain,Ytrain)

	def tahminet(self,model,path):
		img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
		img = cv2.resize(img,(32,32),interpolation=cv2.INTER_CUBIC)
		npimg = np.array(img)
		npimg = np.expand_dims(npimg,axis=2)
		npimg = np.expand_dims(npimg,axis=0)
		print('npimg:',npimg.shape)
		predicts = model.predict(npimg)
		#print('predict',predicts)
		return predicts.tolist()

	def sonuc(self,predictlist):
		print(np.array(predictlist))
		for select in range(len(self.kinds)):
			if predictlist[0][select]==1.0:
				print('select:',select)
				print("this is a %s"%self.kinds[select])
                
                




	def getDictionary(self):
		return self.dictionary


	def getpath(self,types='set'):
		if types == 'all':
			return self.allpath
		elif types=='set':
			return self.setpath
		elif types=='conn':
			return self.connpath
		elif types=='array':
			return self.patharray
	
	def getarray(self):
		return self.picarray
				
	def getlabels(self):
		return self.labels

