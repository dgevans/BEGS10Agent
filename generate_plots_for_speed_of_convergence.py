""" This file generates the plots the conditional mean simulations.
"""
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
import calibrate_begs_id_nu_bgp as Para
data_simulation={}


T=len(Gamma)
N=len(y[0])
T=T-1000


def Transfers(y,Y,t):
    T=(y[t][0,Para.indx_y['c']]-y[t][0,Para.indx_y['w_e']]*y[t][0,Para.indx_y['l']]*(1-Y[t][Para.indx_Y['tau']]))
    gdp=sum(y[t][:,Para.indx_y['l']]*y[t][:,Para.indx_y['w_e']])/N

    return T/gdp


def GovernmentDebt(y):
    a = y[:,Para.indx_y['a']]
    atild = a[1:] - a[0]
    output=sum(y[:,Para.indx_y['l']]*y[:,Para.indx_y['w_e']])
    debt_gdp=sum(atild)/output

    return debt_gdp


def Returns(y,t):
    Uc_ = Para.psi/y[t-1][1,Para.indx_y['c']] 
    UcP=y[t][1,Para.indx_y['UcP']]
    Uc= Para.psi/y[t][1,Para.indx_y['c']] 
    P=UcP/Uc
    kappa_=y[t][1,Para.indx_y['kappa_']]
    rho1_=y[t][1,Para.indx_y['rho1_']]
    EUcp=kappa_/rho1_
    q=Para.beta*EUcp/Uc_
    return P/q
    

def get_chi(y,Y,t):
    UcP=y[t][1,Para.indx_y['UcP']]
    Uc= Para.psi/y[t][1,Para.indx_y['c']] 
    P=UcP/Uc
    Theta=Y[t][Para.indx_Y['Theta']]
    eps=Theta
    chi=(P-1)/eps
    return chi
    
max_ex=7

f,(ax1,ax2) =plt.subplots(2,1,sharex='col')

for ex in [0,1,2,3,4,5,6]:    
    data_simulation= pickle.load(open('data_simulation_cm'+str(int(ex))+'.pickle'))  
    Gamma,Z,Y,Shocks,y=data_simulation    
    T=len(Gamma)-1
    lines_taxes=ax2.plot(np.vstack(np.array(map(lambda t: Y[t][3],range(0,T-2,5))).T))
    lines_debt=ax1.plot(np.vstack(np.array(map(lambda t: GovernmentDebt(y[t]),range(0,T-2,5))).T))
    ax2.set_title(r'taxes-rates')
    ax1.set_title(r'debt-gdp ratio')
    
    plt.setp(lines_taxes[0],linewidth=1.5,color='k',linestyle=':')
    plt.setp(lines_debt[0],linewidth=1.5,color='k',linestyle=':')
    if ex>3:
        plt.setp(lines_taxes[0],linewidth=1+(ex-3)*1.5,color='k',linestyle='-')
        plt.setp(lines_debt[0],linewidth=1+(ex-3)*1.5,color='k',linestyle='-')
    elif ex<3:
        plt.setp(lines_taxes[0],linewidth=1+(ex)*1.5,color='k',linestyle='--')
        plt.setp(lines_debt[0],linewidth=1+(ex)*1.5,color='k',linestyle='--')

     

        
    plt.xlabel('t')    
    ax1.locator_params(nbins=6)
    ax2.locator_params(nbins=5)   
    ax1.autoscale(enable=None, axis='x',tight=True)
    plt.tight_layout()

    plt.savefig('speed_of_convergence.png',dpi=300)        

