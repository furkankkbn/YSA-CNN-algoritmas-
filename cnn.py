import kodlar as cnnlib
from keras.models import Sequential
import numpy as np

Xtrain = None
Ytrain = None
model = None

CNN = cnnlib.CNNmodel('TestData')  #klasorden veri çekliyor
CNN.baglanti()
CNN.veriOku()
#print('picarray:',len(CNN.getarray()))
#print('change:',CNN.tipdegistir())
CNN.sozlukayarla()
#print('dictionary:',CNN.getDictionary())
CNN.jsonpaket()
#CNN.jsonyukle()
CNN.normalize()
CNN.one_hot()

Xtrain = CNN.getarray()
Ytrain = CNN.getlabels()

Xtrain = np.expand_dims(Xtrain,3)

print(Xtrain.shape)

model = Sequential()
model = CNN.modelolustur(model)

print(CNN.ozetal(model))

model = CNN.train(model,Xtrain,Ytrain)

print(CNN.degerlendirme(model,Xtrain,Ytrain))
path = "C:\\img\\moto3.jpg"

ans = CNN.tahminet(model,path)

CNN.sonuc(ans)