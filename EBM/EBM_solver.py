# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 10:40:07 2026

@author: Konstanto
"""

import numpy as np

#An Euler numerical scheme for the computation of the S,E,I,R populations of a SEIR model with equal birth/death rates m
#with transmission rate b (avg contacts/person/time), recovery rate g and latency frequency sig for a period
#of T time-stamps discretized into ts time ticks for a total population N=S+E+I+R with starting populations
#S0,E0,I0,R0. The output is a 5 x ts matrix with each row containing the corresponding compartment's values per time tick 
def Epidemics_SEIR_Euler(b,g,sig,m,N,T,ts,S0,E0,I0,R0):
    t=np.linspace(0,T,ts)
    h=t[1]-t[0]
    y=np.zeros([ts,5])
    y[0]=[0,S0,E0,I0,R0]
    for i in range(1,ts):
        y[i][0]=i#time-stamps
        y[i][1]=y[i-1][1]+h*(m*(N-y[i-1][1])-b*y[i-1][1]*y[i-1][3]/N)#Susceptible
        y[i][2]=y[i-1][2]+h*(b*y[i-1][1]*y[i-1][3]/N-(m+sig)*y[i-1][2])#Exposed
        y[i][3]=y[i-1][3]+h*(sig*y[i-1][2]-(m+g)*y[i-1][3])#Infected
        y[i][4]=y[i-1][4]+h*(g*y[i-1][3]-m*y[i-1][4])#Recovered
    u=np.transpose(y)
    return u
#A Heun (predictor-corrector) numerical scheme for the computation of the S,E,I,R populations of a SEIR model with equal birth/death rates m
#with transmission rate b (avg contacts/person/time), recovery rate g and latency frequency sig for a period
#of T time-stamps discretized into ts time ticks for a total population N=S+E+I+R with starting populations
#S0,E0,I0,R0. The output is a 5 x ts matrix with each row containing the corresponding compartment's values per time tick 
def Epidemics_SEIR_Heun(b,g,sig,m,N,T,ts,S0,E0,I0,R0):
    t=np.linspace(0,T,ts)
    h=t[1]-t[0]
    y=np.zeros([ts,5])
    y[0]=[0,S0,E0,I0,R0]
    for i in range(1,ts):
      y[i][0]=i#time-stamps
      #predictor
      s_predict=y[i-1][1]+h*(m*(N-y[i-1][1])-b*y[i-1][1]*y[i-1][3]/N)#Susceptible
      e_predict=y[i-1][2]+h*(b*y[i-1][1]*y[i-1][3]/N-(m+sig)*y[i-1][2])#Exposed
      i_predict=y[i-1][3]+h*(sig*y[i-1][2]-(m+g)*y[i-1][3])#Infected
      r_predict=y[i-1][4]+h*(g*y[i-1][3])#Recovered
      #corrector
      y[i][1]=y[i-1][1]+h*(m*(N-y[i-1][1])-b*y[i-1][1]*y[i-1][3]/N+m*(N-s_predict)-b*s_predict*i_predict/N)/2#Susceptible
      y[i][2]=y[i-1][2]+h*(b*y[i-1][1]*y[i-1][3]/N-(m+sig)*y[i-1][2]+b*s_predict*i_predict/N-(m+sig)*e_predict)/2#Exposed
      y[i][3]=y[i-1][3]+h*(sig*y[i-1][2]-(m+g)*y[i-1][3]+sig*e_predict-(m+g)*i_predict)/2#Infected
      y[i][4]=y[i-1][4]+h*(g*y[i-1][3]-m*y[i-1][4]+g*i_predict-m*r_predict)/2#Recovered
    u=np.transpose(y)
    return u
