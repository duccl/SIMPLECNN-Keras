import os
import cv2
from random import shuffle,seed
import numpy as np

seed(1)

paths = {'dogs_train':'dataset//training_set//dogs//',
             'cats_train':'dataset//training_set//cats//',
             'dogs_test':'dataset//test_set//dogs//',
             'cats_test':'dataset//test_set//cats//'}

def randomize(x,y):
   link = list(zip(x,y))
   shuffle(link)
   x,y = zip(*link)
   return x,y

def create_lists():
   return [],[],[],[]

def fill(arquivo,rotulo,x,y):
   imagem = cv2.imread(arquivo)
   imagem = cv2.cvtColor(imagem,cv2.COLOR_BGR2RGB)
   imagem = cv2.resize(imagem,(64,64))
   x.append(imagem)
   y.append(rotulo)


def init(quantites_train = 2000,quantites_test = 1000):
   X_treino,X_teste,y_treino,y_teste = create_lists()

   list_dogs_train = os.listdir(paths['dogs_train'])
   list_cats_train = os.listdir(paths['cats_train'])

   list_dogs_test = os.listdir(paths['dogs_test'])
   list_cats_test = os.listdir(paths['cats_test'])

   print('Preenchendo treino\n')
   
   for i in range(quantites_train):
      archive_dog = paths['dogs_train']+list_dogs_train[i]
      archive_cat = paths['cats_train']+list_cats_train[i]
      fill(archive_dog,1,X_treino,y_treino)
      fill(archive_cat,0,X_treino,y_treino)
      
   X_treino,y_treino = randomize(X_treino,y_treino)
   
   print('Preenchendo teste\n')
   
   for j in range(quantites_test):
      archive_dog = paths['dogs_test']+list_dogs_test[j]
      archive_cat = paths['cats_test']+list_cats_test[j]
      fill(archive_dog,1,X_teste,y_teste)
      fill(archive_cat,0,X_teste,y_teste)      

   X_teste,y_teste = randomize(X_teste,y_teste)

   X_teste = np.array(X_teste)/255 #normalizing data
   X_treino = np.array(X_treino)/255 #normalizing data

   y_teste = np.array(y_teste).reshape(-1,1)
   y_treino = np.array(y_treino).reshape(-1,1)
   
   return X_treino,X_teste,y_treino,y_teste
