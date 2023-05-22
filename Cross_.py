#!/usr/bin/env python
# coding: utf-8

# # Evaluating the model using cross_validation

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, activation
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold


# In[2]:


x = pd.read_csv("https://raw.githubusercontent.com/PacktWorkshops/The-Deep-Learning-with-Keras-Workshop/master/Chapter03/data/HCV_feats.csv")
y = pd.read_csv("https://raw.githubusercontent.com/PacktWorkshops/The-Deep-Learning-with-Keras-Workshop/master/Chapter03/data/HCV_target.csv")


# In[3]:


print(f"Number of features: {x.shape[1]}")
print(f"Number of examples: {x.shape[0]}")
print(f"Number of features: {np.unique(y)}")


# In[4]:


def build_model():
    model = Sequential()
    model.add(Dense(4,input_dim=x.shape[1],activation = 'tanh'))
    model.add(Dense(2,activation = 'tanh'))
    model.add(Dense(1,activation = 'sigmoid'))
    model.compile(loss="binary_crossentropy",metrics="accuracy",optimizer="adam")
    return model
    



# In[5]:


sc = StandardScaler()
x = pd.DataFrame(sc.fit_transform(x),columns=x.columns)
x


# In[6]:


np.random.seed(1)

classifier = KerasClassifier(build_fn=build_model,epochs = 100,verbose=0, batch_size=20,shuffle=False)
cv = StratifiedKFold(n_splits=5,shuffle=False)
cv_score= cross_val_score(classifier,x,y,cv=cv,verbose=0)



# In[7]:


for f in range(5):
    print("Test accuracy af folder", f+1, "=", cv_score[f])
print("\n")
print("final cross validation result", cv_score.mean())
print("standard deviation result", cv_score.std())


# # Improve the classifier

# In[8]:


def Build_model_1():
    model =Sequential()
    model.add(Dense(4,input_dim=x.shape[1],activation = 'relu'))
    model.add(Dense(4,activation = 'relu'))
    model.add(Dense(4,activation = 'relu'))
    model.add(Dense(1,activation="sigmoid"))
    model.compile(loss="binary_crossentropy",optimizer = 'adam',metrics="accuracy")
    return model

def Build_model_2():
    model =Sequential()
    model.add(Dense(4,input_dim=x.shape[1],activation = 'relu'))
    model.add(Dense(2,activation = 'relu'))
    model.add(Dense(1,activation="sigmoid"))
    model.compile(loss="binary_crossentropy",optimizer = 'adam',metrics="accuracy")
    return model

def Build_model_3():
    model =Sequential()
    model.add(Dense(8,input_dim=x.shape[1],activation = 'relu'))
    model.add(Dense(8,activation = 'relu'))
    model.add(Dense(1,activation="sigmoid"))
    model.compile(loss="binary_crossentropy",optimizer = 'adam',metrics="accuracy")
    return model


# In[9]:


models = [Build_model_1,Build_model_2,Build_model_3]
results=[]
np.random.seed(42)

for m in range(len(models)):
    my_model = KerasClassifier(build_fn=models[m],epochs=100,verbose=0,batch_size=20,shuffle=False)
    cv=KFold(n_splits=5)
    result = cross_val_score(my_model,x,y,cv=cv)
    results.append(result)


# In[10]:


for s in range(len(results)):
    print("Total mean accuracy:", abs(results[s].mean()))



# In[11]:


epochs=[100,200]
batches=[10,20]
results_1=[]
for e in range(len(epochs)):
    for b in range(len(batches)):
        my_model = KerasClassifier(build_fn=Build_model_1,epochs=epochs[e],batch_size=batches[b],verbose=0,shuffle=False)
        cv=KFold(n_splits=5)
        result = cross_val_score(my_model,x,y,cv=cv)
        results_1.append(result)



# In[12]:


r=0
for e in range(len(epochs)):
    for b in range(len(batches)):
        print("epochs:", epochs[e], "Batches:", batches[b], "Total accuracy function:", abs(results_1[r].mean()))
        r+=1


# In[13]:


def Build_model_1(activation='relu',optimizer= 'adam'):
    model =Sequential(
    (Dense(4,input_dim=x.shape[1],activation = activation))
    (Dense(4,activation = activation))
    (Dense(4,activation = activation))
    (Dense(1,activation=activation)))
    model.compile(loss="binary_crossentropy",optimizer = optimizer,metrics="accuracy")
    return model

optimizers = ['rmsprop','adam','sgd']
activations= ['relu','tanh']
results_2=[]

for o in range(len(optimizers)):
    for a in range(len(activations)):
        optimizer = optimizers[o]
        activation = activations[a]
        my_model = KerasClassifier(build_fn=Build_model_1,epochs=100,batch_size=20,verbose=0,shuffle=False)
        cv= KFold(n_splits=5)
        result= cross_val_score(my_model,x,y,cv=cv)
        results_2.append(result)



# In[14]:


w=0
for o in range(len(optimizers)):
    for a in range(len(activations)):
        print("optimizer:", optimizers[o],"activations:", activations[a],"Total accuracy:", abs(results_2[w].mean()))
        w+=1


# # Model selection

# In[15]:


x= pd.read_csv("https://raw.githubusercontent.com/PacktWorkshops/The-Deep-Learning-with-Keras-Workshop/master/Chapter04/data/traffic_volume_feats.csv")
y= pd.read_csv("https://raw.githubusercontent.com/PacktWorkshops/The-Deep-Learning-with-Keras-Workshop/master/Chapter04/data/traffic_volume_target.csv")


# In[16]:


print("numer of examples:", x.shape[0])
print("number of feature:",x.shape[1])
print("range of the output:",[y.min(),y.max()])


# In[17]:


from keras.wrappers.scikit_learn import KerasRegressor


# In[18]:


def model_1(optimizer='adam'):
    model=Sequential()
    model.add(Dense(10,input_dim=x.shape[1],activation="relu"))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error",optimizer=optimizer)
    return model
def model_2(optimizer='adam'):
    model=Sequential()
    model.add(Dense(10,input_dim=x.shape[1],activation="relu"))
    model.add(Dense(10,activation="relu"))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error",optimizer=optimizer)
    return model

def model_3(optimizer='adam'):
    model=Sequential()
    model.add(Dense(10,input_dim=x.shape[1],activation="relu"))
    model.add(Dense(10,activation="relu"))
    model.add(Dense(10,activation="relu"))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error",optimizer=optimizer)
    return model


# In[19]:


scl= StandardScaler()
x=scl.fit_transform(x)
from tensorflow import random


# In[ ]:


np.random.seed(1)
random.set_seed(1)
models=[model_1,model_2,model_3]
results=[]
for m in range(len(models)):
    regressor = KerasRegressor(build_fn=models[m],epochs = 100, batch_size=5,verbose=0,shuffle=False)
    cv=KFold(n_splits=5)
    resulet = cross_val_score(regressor,x,y,cv=cv)
    results.append(resulet)


# In[ ]:


results


# In[ ]:


modls=['model_1','model_2','model_3']
for m in range(len(models)):
    print("Model Number:", modls[m],"The total loss function:", abs(results[m].mean()))



# # Regularization (L1/L2)

# In[ ]:


import pandas as pd
import numpy as np
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow import random
from keras.layers import Dense, activation
from keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt
from keras.regularizers import l2


# In[ ]:


x =pd.read_csv("https://raw.githubusercontent.com/PacktWorkshops/The-Deep-Learning-with-Keras-Workshop/master/Chapter05/data/avila-tr_feats.csv")
y= pd.read_csv("https://raw.githubusercontent.com/PacktWorkshops/The-Deep-Learning-with-Keras-Workshop/master/Chapter05/data/avila-tr_target.csv")


# In[ ]:


seed=32
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=seed)


# In[ ]:


np.random.seed(seed)
random.set_seed(seed)

model =Sequential()
model.add(Dense(10, input_dim=x.shape[1], activation='relu'))
model.add(Dense(6,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='sgd',metrics='accuracy')


# In[ ]:


np.random.seed(seed)
random.set_seed(seed)
history=model.fit(x_train,y_train,epochs=100,batch_size=20,validation_data=(x_test,y_test),verbose=0,shuffle=False)


# In[ ]:


np.random.seed(seed)
random.set_seed(seed)
plt.plot(history.history['loss'],label="loss")
plt.plot(history.history['val_loss'],label="val_loss")
plt.ylim(0,1)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()


# In[ ]:


print("best accuracy :",max(history.history["accuracy"]))


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2
np.random.seed(seed)
random.set_seed(seed)
model_2 = Sequential()
model_2.add(Dense(10, input_dim=x.shape[1], activation='relu', kernel_regularizer=l2(0.01)))
model_2.add(Dense(6, activation='relu', kernel_regularizer=l2(0.01)))
model_2.add(Dense(4, activation='relu', kernel_regularizer=l2(0.01)))
model_2.add(Dense(1, activation='sigmoid'))
model_2.compile(loss='binary_crossentropy',optimizer='sgd',metrics='accuracy')
history=model_2.fit(x_train,y_train,epochs=100,batch_size=20,validation_data=(x_test,y_test),verbose=0,shuffle=False)

plt.plot(history.history['loss'],label="loss")
plt.plot(history.history['val_loss'],label="val_loss")
plt.ylim(0,1)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()

print("best accuracy :",max(history.history["val_accuracy"]))


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2
np.random.seed(seed)
random.set_seed(seed)
model_3 = Sequential()
model_3.add(Dense(10, input_dim=x.shape[1], activation='relu', kernel_regularizer=l2(0.1)))
model_3.add(Dense(6, activation='relu', kernel_regularizer=l2(0.1)))
model_3.add(Dense(4, activation='relu', kernel_regularizer=l2(0.1)))
model_3.add(Dense(1, activation='sigmoid'))
model_3.compile(loss='binary_crossentropy',optimizer='sgd',metrics='accuracy')
history=model_3.fit(x_train,y_train,epochs=100,batch_size=20,validation_data=(x_test,y_test),verbose=0,shuffle=False)

plt.plot(history.history['loss'],label="loss")
plt.plot(history.history['val_loss'],label="val_loss")
plt.ylim(0,1)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()

print("best accuracy :",max(history.history["val_accuracy"]))


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2
np.random.seed(seed)
random.set_seed(seed)
model_4 = Sequential()
model_4.add(Dense(10, input_dim=x.shape[1], activation='relu', kernel_regularizer=l2(0.005)))
model_4.add(Dense(6, activation='relu', kernel_regularizer=l2(0.005)))
model_4.add(Dense(4, activation='relu', kernel_regularizer=l2(0.005)))
model_4.add(Dense(1, activation='sigmoid'))
model_4.compile(loss='binary_crossentropy',optimizer='sgd',metrics='accuracy')
history=model_4.fit(x_train,y_train,epochs=100,batch_size=20,validation_data=(x_test,y_test),verbose=0,shuffle=False)

plt.plot(history.history['loss'],label="loss")
plt.plot(history.history['val_loss'],label="val_loss")
plt.ylim(0,1)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()

print("best accuracy :",max(history.history["val_accuracy"]))


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l1_l2
np.random.seed(seed)
random.set_seed(seed)
l1_pa=0.005
l2_pa=0.005
model_5 = Sequential()
model_5.add(Dense(10, input_dim=x.shape[1], activation='relu', kernel_regularizer=l1_l2(l1=l1_pa,l2=l2_pa)))
model_5.add(Dense(6, activation='relu', kernel_regularizer=l1_l2(l1=l1_pa,l2=l2_pa)))
model_5.add(Dense(4, activation='relu', kernel_regularizer=l1_l2(l1=l1_pa,l2=l2_pa)))
model_5.add(Dense(1, activation='sigmoid'))
model_5.compile(loss='binary_crossentropy',optimizer='sgd',metrics='accuracy')
history=model_5.fit(x_train,y_train,epochs=100,batch_size=20,validation_data=(x_test,y_test),verbose=0,verbose=0,shuffle=False)


plt.plot(history.history['loss'],label="loss")
plt.plot(history.history['val_loss'],label="val_loss")
plt.ylim(0,1)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()

print("best accuracy :",max(history.history["val_accuracy"]))


# # Regularization (Dropout)

# In[ ]:


import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, activation,Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot  as plt


# In[ ]:


x = pd.read_csv("https://raw.githubusercontent.com/PacktWorkshops/The-Deep-Learning-with-Keras-Workshop/master/Chapter05/data/traffic_volume_feats.csv")
y= pd.read_csv("https://raw.githubusercontent.com/PacktWorkshops/The-Deep-Learning-with-Keras-Workshop/master/Chapter05/data/traffic_volume_target.csv")


# In[ ]:


seed=1
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=seed)


# In[ ]:


np.random.seed(seed)
random.set_seed(seed)

model =Sequential()
model.add(Dense(10,input_dim=x_train.shape[1],activation ="relu"))
model.add(Dense(10,activation ="relu"))
model.add(Dense(1))
model.compile(loss="mean_squared_error",optimizer="rmsprop")

history= model.fit(x_train,y_train,epochs=200,batch_size=50,verbose=0,validation_data=(x_test,y_test),shuffle=False)


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(("loss","vla_loss"))

plt.xlabel("epochs")
plt.ylabel('loss')
print("lowest error on training set=",min(history.history["loss"]))
print("lowest error on test set=",min(history.history["val_loss"]))


# In[ ]:


np.random.seed(seed)
random.set_seed(seed)

model_1 =Sequential()
model_1.add(Dense(10,input_dim=x_train.shape[1],activation ="relu"))
model_1.add(Dropout(0.1))
model_1.add(Dense(10,activation ="relu"))
model_1.add(Dense(1))
model_1.compile(loss="mean_squared_error",optimizer="rmsprop")

history_1= model_1.fit(x_train,y_train,epochs=200,batch_size=50,verbose=0,validation_data=(x_test,y_test),shuffle=False)

plt.plot(history_1.history['loss'])
plt.plot(history_1.history['val_loss'])
plt.legend(("loss","vla_loss"))
plt.xlabel("epochs")
plt.ylabel('loss')


# In[ ]:


print("lowest error on training set=",min(history_1.history["loss"]))
print("lowest error on test set=",min(history_1.history["val_loss"]))


# In[ ]:


np.random.seed(seed)
random.set_seed(seed)

model_2 =Sequential()
model_2.add(Dense(10,input_dim=x_train.shape[1],activation ="relu"))
model_2.add(Dropout(0.1))
model_2.add(Dense(10,activation ="relu"))
model_2.add(Dropout(0.1))
model_2.add(Dense(1))
model_2.compile(loss="mean_squared_error",optimizer="rmsprop")

history_2= model_2.fit(x_train,y_train,epochs=200,batch_size=50,verbose=0,validation_data=(x_test,y_test),shuffle=False)

plt.plot(history_2.history['loss'])
plt.plot(history_2.history['val_loss'])
plt.legend(("loss","vla_loss"))
plt.xlabel("epochs")
plt.ylabel('loss')

print("lowest error on training set=",min(history_2.history["loss"]))
print("lowest error on test set=",min(history_2.history["val_loss"]))


# In[ ]:


np.random.seed(seed)
random.set_seed(seed)

model_3 =Sequential()
model_3.add(Dense(10,input_dim=x_train.shape[1],activation ="relu"))
model_3.add(Dropout(0.2))
model_3.add(Dense(10,activation ="relu"))
model_3.add(Dropout(0.1))
model_3.add(Dense(1))
model_3.compile(loss="mean_squared_error",optimizer="rmsprop")

history_3= model_3.fit(x_train,y_train,epochs=200,batch_size=50,verbose=0,validation_data=(x_test,y_test),shuffle=False)

plt.plot(history_3.history['loss'])
plt.plot(history_3.history['val_loss'])
plt.legend(("loss","vla_loss"))
plt.xlabel("epochs")
plt.ylabel('loss')

print("lowest error on training set=",min(history_3.history["loss"]))
print("lowest error on test set=",min(history_3.history["val_loss"]))


# # Regularization (Hyperparameter Tuning)

# In[ ]:


from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasRegressor
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


x =pd.read_csv("https://raw.githubusercontent.com/PacktWorkshops/The-Deep-Learning-with-Keras-Workshop/master/Chapter05/data/avila-tr_feats.csv")
y= pd.read_csv("https://raw.githubusercontent.com/PacktWorkshops/The-Deep-Learning-with-Keras-Workshop/master/Chapter05/data/avila-tr_target.csv")


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2
def build_model(lambda_parameter):
    model = Sequential()
    model.add(Dense(10, input_dim=x.shape[1], activation='relu', kernel_regularizer=l2(lambda_parameter)))
    model.add(Dense(6, activation='relu', kernel_regularizer=l2(lambda_parameter)))
    model.add(Dense(4, activation='relu', kernel_regularizer=l2(lambda_parameter)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model


# In[ ]:


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
from tensorflow import random

seed = 1
np.random.seed(seed)
random.set_seed(seed)
model = KerasClassifier(build_fn=build_model, verbose=0)
lambda_parameter = [0.01, 0.5, 1]
epochs = [50, 100]
batch_size = [20]
param_grid = dict(lambda_parameter=lambda_parameter, epochs=epochs, batch_size=batch_size)
grid_seach = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
results_1 = grid_seach.fit(x, y)


# In[ ]:


print("Best cross validation score=",results_1.best_score_)
print("Parameter Best cross validation error =",results_1.best_params_)
accuracy_means = results_1.cv_results_['mean_test_score']
accuracy_stds = results_1.cv_results_['std_test_score']
parameters = results_1.cv_results_['params']
for p in range(len(parameters)):
    print("Accuracy %f (std %f) for params %r" % (accuracy_means[p], accuracy_stds[p], parameters[p]))


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense

from keras.layers import Dropout
def build_model(rate):
    model = Sequential()
    model.add(Dense(10, input_dim=x.shape[1], activation='relu'))
    model.add(Dropout(rate))
    model.add(Dense(6, activation='relu'))
    model.add(Dropout(rate))
    model.add(Dense(4, activation='relu'))
    model.add(Dropout(rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model


# In[ ]:


np.random.seed(seed)
random.set_seed(seed)

model = KerasClassifier(build_fn=build_model, verbose=0)
rate = [0, 0.1, 0.2]
epochs = [50, 100]
batch_size = [20]
param_grid = dict(rate=rate, epochs=epochs, batch_size=batch_size)
grid_seach = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
results_3 = grid_seach.fit(x, y)


# In[ ]:


print("Best cross validation score=",results_3.best_score_)
print("Parameter Best cross validation error =",results_3.best_params_)
accuracy_means = results_3.cv_results_['mean_test_score']
accuracy_stds = results_3.cv_results_['std_test_score']
parameters = results_3.cv_results_['params']
for p in range(len(parameters)):
    print("Accuracy %f (std %f) for params %r" % (accuracy_means[p], accuracy_stds[p], parameters[p]))


# In[ ]:
