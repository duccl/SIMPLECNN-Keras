from keras.models import model_from_json
from sklearn.metrics import confusion_matrix,roc_curve,auc
from leitor_dogs_cats import init
import matplotlib.pyplot as plt


X_train,X_test,y_train,y_test = init()

archive = open('cnn_rmsprop_5ep.json','r')

model = archive.read()

model = model_from_json(model)

model.load_weights('cnn_rmsprop_5ep.h5')

probabilites = model.predict(X_test)
predicts = model.predict_classes(X_test)
falsePositiveRate,truePositiveRate,thresholds = roc_curve(y_test,probabilites)
auc_score = auc(falsePositiveRate,truePositiveRate)

print(confusion_matrix(y_test,predicts))

plt.plot(falsePositiveRate,truePositiveRate,label = 'AUC Score: '+str(auc_score))
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
