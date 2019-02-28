from leitor_dogs_cats import init,paths
from cnn import cnn_model
import os
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import classification_report,confusion_matrix,roc_curve,auc

X_train,X_test,y_train,y_test = init()
previsoes,classificador = cnn_model(X_train,X_test[:600],y_train,y_test[:600])



# ---------------- Fase de avaliação -------------------
print(classification_report(y_test[:600],previsoes))
print(confusion_matrix(y_test[:600],previsoes))

def le_imagem(arquivo):
   imagem = cv2.imread(arquivo)
   imagem = cv2.cvtColor(imagem,cv2.COLOR_BGR2RGB)
   return imagem

def predict(imagem):
   imagem = cv2.resize(imagem,(64,64))
   imagem = imagem.reshape(1,64,64,3)
   previsao = classificador.predict_classes(imagem)
   if previsao == 1:
      return 'dog'
   return 'cat'

def plot(imagem,label):
   plt.imshow(imagem)
   plt.xlabel("I think it's a "+label)
   plt.xticks([])
   plt.yticks([])
   plt.show()


for j in range(10):
   cachorro = le_imagem(paths['dogs_test']+os.listdir(paths['dogs_test'])[j])
   gato = le_imagem(paths['cats_test']+os.listdir(paths['cats_test'])[j])
   
   previsao_cachorro = predict(cachorro)
   previsao_gato = predict(gato)

   plot(cachorro,previsao_cachorro)
   plot(gato,previsao_gato)

probabilites = classificador.predict(X_test[:600])
falsePositiveRate,truePositiveRate,thresholds = roc_curve(y_test,probabilites)
auc_score = auc(falsePositiveRate,truePositiveRate)

plt.plot(falsePositiveRate,truePositiveRate,label = 'AUC Score: '+str(auc_score))
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
