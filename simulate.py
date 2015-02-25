# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 16:14:02 2014

@author: dgevans
"""

import numpy as np
from IPython.parallel import Client
from IPython.parallel import Reference
c = Client()
v = c[:] 


    
    
def simulate_aggstate(Para,Gamma,Z,Y,Shocks,y,T,T0=0,agg_shocks=None,quadratic = True):
    '''
    Simulates a sequence of state path for a given Para
    '''
    #approximate.calibrate(Para)
    t = T0+1
    v.execute('approximate.shock = None')
    while t< T:
        if agg_shocks is not None:
            v['agg_shock'] = agg_shocks[t]
            v.execute('approximate.shock = agg_shock')
        Gamma[t],Z[t],Y[t-1], Shocks[t-1],y[t-1]= update_state_parallel_aggstate(Para,Gamma[t-1],Z[t-1],quadratic)
        print t,np.exp(Z[t]),Y[t-1][4:6]
        t += 1
    
def update_state_parallel_aggstate(Para,Gamma,Z,quadratic = True):
    '''
    Updates the state using parallel code
    '''
    v.block = True
    v['Gamma'] = Gamma
    v['Z'] = np.array(Z)
    v.execute('approx = approximate.approximate(Gamma)')
    diff = np.inf
    n = 0.
    while diff > 0.001 and n < 1:
        v.execute('data = approx.iterate(Z)')
        Gamma_new_t,ZNew_t,Y_t,Shocks_t,y_t = filter(None,v['data'])[0]
        error = np.linalg.norm(np.mean(Gamma_new_t[:,:2],0))
        Gamma_new,ZNew,Y,Shocks,y = Gamma_new_t,ZNew_t,Y_t,Shocks_t,y_t
        n += 1
    return Para.nomalize(Gamma_new.copy()),ZNew.copy(),Y.copy(),Shocks.copy(),y.copy()
    
    
    
def simulate_aggstate_ConditionalMean(Para,Gamma,Z,Y,Shocks,y,T,T0=0):
    '''
    Simulates a sequence of state path for a given Para
    '''
    #approximate.calibrate(Para)
    t = T0+1
    v.execute('approximate.shock = None')
    while t< T:
        Gamma[t],Z[t],Y[t-1],Shocks[t-1],y[t-1]= update_state_parallel_ConditionalMean(Para,Gamma[t-1],Z[t-1])
        print t,np.exp(Z[t]),Y[t-1][4:6]
        t += 1
    
def update_state_parallel_ConditionalMean(Para,Gamma,Z):
    '''
    Updates the state using parallel code
    '''
    v.block = True
    v['Gamma'] = Gamma
    v['Z'] = np.array(Z)
    v.execute('approx = approximate.approximate(Gamma)')
    diff = np.inf
    n = 0.
    while diff > 0.001 and n < 1:
        v.execute('data = approx.iterate_ConditionalMean(Z)')
        Gamma_new_t,ZNew_t,Y_t,Shocks_t,y_t = filter(None,v['data'])[0]
        Gamma_new,ZNew,Y,Shocks,y = Gamma_new_t,ZNew_t,Y_t,Shocks_t,y_t
        n += 1
    return Para.nomalize(Gamma_new.copy()),ZNew.copy(),Y.copy(),Shocks.copy(),y.copy()
    
    
