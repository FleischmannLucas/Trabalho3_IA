# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 19:03:39 2021

@author: Lucas Gava e Gustavo Souza
"""

from geneticalgorithm import geneticalgorithm as ga
from control import tf, feedback, step_response
import matplotlib.pyplot as plt
import control as crt
import numpy as np

def f(X):                       #Definição da função de custo(ou de avaliação) que serve como
                                #feedback para o algoritimo genético, função a ser minimizada. 
                                
    ft = tf([2], [4, 1])        #Função de transferencia do sistema
    
    ftp = tf([X[0]],[1])            #Função de transferencia controlador proporcional (Kp)
    fti = tf([X[1]],[1, 0])         #Função de transferencia controlador integral (Ki/s)
    ftd = tf([X[2], 0],[1])         #Função de transferencia controlador derivativo (Ki*s)
    
    prl = crt.parallel( ftp, fti, ftd)    #Concatena o PID em paralelo
    ser = crt.series(ft, prl)             #Liga o PID e a FT do sistemma em série 
    
    sis  =  feedback(ser, 1)        #Realimentação unitaria    
    
    t = np.linspace(0, 20, 201)     #Tempo de simulação = 5*Constante de tempo(4) = 20s
    _, y1 = step_response(sis, t)   #Resposta ao degral do sistema com PID
    
    z = np.array((y1 - 1))      #Array do erro                
    z2 = z*z                    #Erro ao quadrado, elimina valores negativos
    erro = sum(z2*t)          #Integral do erro quadrático ISE 

    return erro
    


varbound = np.array([[0,20]]*3)             #Limite das variáveis 
    

#Define os parametros de teste do Algorimo Genético
algorithm_param = {'max_num_iteration': 1000,\
                   'population_size': 100,\
                   'mutation_probability':0.1,\
                   'elit_ratio': 0.01,\
                   'crossover_probability': 0.5,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':None}
    
#inicia o modelo do Algorimo Genético
model=ga(function=f ,dimension= 3 , variable_type='int',variable_boundaries=varbound,algorithm_parameters=algorithm_param) 

#Roda o Algorimo Genético
model.run()  

X = model.best_variable             #registra a melhor resposta
erro = model.best_function          #registra o menor valor da funçao objetivo, menor ISE


#Simulação dos resultados e plotagem dos graficos
ft = tf([2], [4, 1])       
    
ftp = tf([X[0]],[1])           
fti = tf([X[1]],[1, 0])         
ftd = tf([X[2], 0],[1])        
    
prl = crt.parallel( ftp, fti, ftd)   
ser = crt.series(ft, prl)            
    
sis  =  feedback(ser, 1)  

t = np.linspace(0, 20, 20000)
_,g1 = step_response(sis, t)        #gráfico resposta ao degrau, sistema com PID    
_,g2 = step_response(ft, t)         #gráfico resposta ao degrau, malha aberta sem PID


plt.figure(1)
plt.plot(t, g1)
plt.ylim([-0.1, 2])
plt.xlim([0, 20])
plt.xlabel('Tempo(s)')
plt.ylabel('Saída')
plt.title('Resposta ao degrau do Sistema com PID')


plt.figure(2)
plt.plot(t, g2)
plt.ylim([-0.1, 3])
plt.xlim([ 0, 20])
plt.xlabel('Tempo(s)')
plt.ylabel('Saída')
plt.title('Resposta ao degrau do Sistema sem PID')
    
    
