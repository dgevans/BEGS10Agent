# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 16:14:02 2014

@author: dgevans
"""

approximate = None
import numpy as np


def simulate_aggstate(Para,Gamma,Z,Y,Shocks,y,T,T0=0,agg_shocks=None,quadratic = True):
    '''
    Simulates using the MPI code rather than Ipython parallel
    '''
    approximate.calibrate(Para)
    t = T0+1
    while t < T:
        if agg_shocks is not None:
            approximate.shock = agg_shocks[t]
        
            
        Gamma[t],Z[t],Y[t-1], Shocks[t-1],y[t-1]= update_state_parallel_aggstate(Para,Gamma[t-1],Z[t-1],quadratic)
        print 'iteration '+str(t), Y[t-1][3]
        t += 1
        
def simulate_aggstate_ConditionalMean(Para,Gamma,Z,Y,Shocks,y,T,T0=0,agg_shocks=None,quadratic = True):
    '''
    Simulates using the MPI code rather than Ipython parallel
    '''
    approximate.calibrate(Para)
    t = T0+1
    while t < T:
        if agg_shocks is not None:
            approximate.shock = agg_shocks[t]
        
        Gamma[t],Z[t],Y[t-1], Shocks[t-1],y[t-1]= update_state_parallel_aggstate_ConditionalMean(Para,Gamma[t-1],Z[t-1])
        print t,Y[t-1][3]
        t += 1


    
def update_state_parallel_aggstate(Para,Gamma,Z,Y0guess,quadratic = True):
    '''
    Updates the state using parallel code
    '''
    Z = np.array(Z)
    approx = approximate.approximate(Gamma  )
    data = approx.iterate(Z)
    Gamma_new,ZNew,Y,Shocks,y = data
    return Para.nomalize(Gamma_new),ZNew,Y,Shocks,y
    
    
def update_state_parallel_aggstate_ConditionalMean(Para,Gamma,Z):
    '''
    Updates the state using parallel code
    '''
    Z = np.array(Z)
    approx = approximate.approximate(Gamma)
    data = approx.iterate_ConditionalMean(Z)
    Gamma_new,ZNew,Y,Shocks,y = data
    return Para.nomalize(Gamma_new),ZNew,Y,Shocks,y
    