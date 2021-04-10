# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 21:13:35 2021

@author: Gustavo Souza e Lucas Gava
"""

import numpy as np
from geneticalgorithm import geneticalgorithm as ga
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split   #nos permite dividir a entrada em grupos
from sklearn.metrics import accuracy_score 


#Carrega o iris dataset em iris 
iris = load_iris()


#divisao do data set entra grupo de treino e test
cetosa= iris.data[:50,:]
cetosay= iris.target[:50]
cet=np.array(cetosa)
cety=np.array(cetosay)
cettrain, cettest, cetytrain, cetytest=train_test_split(cet, cety, random_state=5)

versicolor= iris.data[50:100,:]
versicolory= iris.target[50:100]
ver=np.array(versicolor)
very=np.array(versicolory)
vertrain, vertest, verytrain, verytest=train_test_split(ver, very, random_state=5)

virginica= iris.data[100:150,:]
virginicay= iris.target[100:150]
virg=np.array(virginica)
virgy=np.array(virginicay)
virgtrain, virgtest, virgytrain, virgytest=train_test_split(virg, virgy, random_state=5)

Xtest = np.concatenate((cettest, vertest, virgtest), axis=0)
Xtrain = np.concatenate((cettrain, vertrain, virgtrain), axis=0)
Ytest = np.concatenate((cetytest, verytest, virgytest), axis=0)
Ytrain = np.concatenate((cetytrain, verytrain, virgytrain), axis=0)


def f(X):                       #Definição da função de custo(ou de avaliação) que serve como
                                #feedback para o algoritimo genético  
    I=X[0]
    
    I= I.astype(np.int64)   #Entrada para o numero de visinhos N deve ser inteiro    
    
    #Implementa o Algoritmo KNN
    neigh = KNeighborsClassifier(n_neighbors = I , weights="uniform")
    neigh.fit(Xtrain, Ytrain)

    #Prevendo novos valores
    Y = neigh.predict(Xtest)
    z = accuracy_score( Y, Ytest)       #Z será o índice de acertos para determinado N
    
    return -z 


varbound = np.array([[1,len(Xtrain)]])  #Limites das variáveis como o parametro N do algorimo KNN pode 
                                        #variar entre 1 e o numero de individuos na popilação de test       


#Define os parametros de teste do Algorimo Genético

algorithm_param = {'max_num_iteration': 15,\
                   'population_size': 111,\
                   'mutation_probability':0.1,\
                   'elit_ratio': 0.01,\
                   'crossover_probability': 0.5,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':None}
    

#inicia o modelo do Algorimo Genético
model=ga(function=f ,dimension= 1 , variable_type='int',variable_boundaries=varbound,algorithm_parameters=algorithm_param) 

#Roda o Algorimo Genético
model.run()  
    
    
    
    
    
    