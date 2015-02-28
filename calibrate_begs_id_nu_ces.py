# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 12:18:27 2014

@author: dgevans
"""
import numpy as np

beta = 0.98
sigma_e = np.array([0.0,0.])
sigma_E = 0.03
chi = 0.0
ineq_slope=1.0/0.8
tfp_level=1.
ll = np.array([-np.inf,-np.inf,-np.inf])
ul = np.array([np.inf,np.inf,np.inf])
bgp_flag=0
n = 2 # number of measurability constraints
nG = 2 # number of aggregate measurability constraints.
ny = 15 # number of individual controls (m_{t},mu_{t},c_{t},l_{t},rho1_,rho2,phi,x_{t-1},kappa_{t-1}) Note that the forward looking terms are at the end
ne = 4 # number of Expectation Terms (E_t u_{c,t+1}, E_t u_{c,t+1}mu_{t+1} E_{t}x_{t-1 E_t rho_{1,t-1}} [This makes the control x_{t-1},rho_{t-1} indeed time t-1 measuable])
nY = 7 # Number of aggregates (alpha_1,alpha_2,tau,eta,lambda)
nz = 5 # Number of individual states (m_{t-1},mu_{t-1})
nv = 2 # number of forward looking terms (x_t,rho1_t)
n_p = 1 #number of parameters
nZ = 1
neps = len(sigma_e)

phat = np.array([-0.00])

indx_y = dict(zip('logm,muhat,e,omega,I,c,l,rho1_,rho2,phi,w_e,UcP,a,x_,kappa_'.split(','), range(ny))) #Yay python string manipulation

indx_Y = dict(zip('Theta,alpha1,alpha2,tau,eta,lamb,T'.split(','), range(nY)))

def F(w):
    '''
    Individual first order conditions
    '''
    logm,muhat,e,omega,I,c,l,rho1_,rho2,phi,w_e,UcP,a,x_,kappa_ = w[:ny] #y
    EUcP,EUc_muP,Ex_,Erho1_ = w[ny:ny+ne] #e
    Theta,alpha1,alpha2,tau,eta,lamb,T = w[ny+ne:ny+ne+nY] #Y
    logm_,muhat_,e_,omega_,I_= w[ny+ne+nY:ny+ne+nY+nz] #z
    x,kappa = w[ny+ne+nY+nz:ny+ne+nY+nz+nv] #v
    nu = w[ny+ne+nY+nz+nv+n_p-1] #v
    Theta_ = w[ny+ne+nY+nz+nv+n_p+nZ-1] #Z
    eps_p,eps_t = w[ny+ne+nY+nz+nv+n_p+nZ:ny+ne+nY+nz+nv+n_p+nZ+neps] #shock
    Eps = w[ny+ne+nY+nz+nv+n_p+nZ+neps] #aggregate shockaggregate shock
    
    m_,m = np.exp(logm_),np.exp(logm)
    mu_ = muhat_ * m_
    mu = muhat * m    
    
    P = 1. + chi*Eps #payoff shock
    
    Uc = c**(-sigma)
    Ucc = -sigma*c**(-sigma-1)
    Ul = -l**(gamma)
    Ull = -gamma*l**(gamma-1)
    
    
    
    ret = np.empty(ny+n,dtype=w.dtype)
    ret[0] = x_ - Ex_ # x is independent of eps
    ret[1] = rho1_ - Erho1_    
    ret[2] = x_*UcP/(beta*EUcP) - Uc*(c-T) - Ul*l - x #impl
    ret[3] = alpha2 - m*Uc #defines m
    ret[4] = (1-tau)*w_e*Uc + Ul #wage
    ret[5] = omega*Ul - mu*(Ull*l + Ul) - phi*Ull + lamb*w_e
    ret[6] = rho2 + kappa/Uc + eta/Uc
    ret[7] = omega*Uc + x_*Ucc*P/(beta*EUcP)*(mu-mu_) - mu*(Ucc*(c-T) + Uc) + rho1_*m_*Ucc/beta \
             +rho2*m*Ucc - phi*w_e*(1-tau)*Ucc - lamb
    ret[8] = kappa_ - rho1_*EUcP
    ret[9] = e - nu*e_ - eps_p
    ret[10] = w_e - np.exp(e+eps_t+Theta*( tfp_level+(0.8-I)*ineq_slope) )
    ret[11] = UcP - Uc*P
    ret[12] = omega-omega_
    ret[13] = a - x/Uc
    ret[14] = I - I_
    ret[15] = mu_*EUcP - EUc_muP
    ret[16] = alpha1 - m_*EUcP # bond pricing
    
    return ret
    
def G(w):
    '''
    Aggregate equations
    '''
    logm,muhat,e,omega,I,c,l,rho1_,rho2,phi,w_e,UcP,a,x_,kappa_ = w[:ny] #y
    Theta,alpha1,alpha2,tau,eta,lamb,T = w[ny+ne:ny+ne+nY] #Y
    Theta_ = w[ny+ne+nY+nz+nv+n_p+nZ-1] #Z
    eps_p,eps_t = w[ny+ne+nY+nz+nv+n_p+nZ:ny+ne+nY+nz+nv+n_p+nZ+neps] #shock
    Eps = w[ny+ne+nY+nz+nv+n_p+nZ+neps] #aggregate shockaggregate shock
 
    m = np.exp(logm)
    Uc = c**(-sigma)
    
    ret = np.empty(nY+nG,dtype=w.dtype)
    ret[0] = alpha1 #alpha_1 can't depend on Eps
    ret[1] = muhat # muhat must integrate to zero for all Eps    
    ret[2] = 0 - logm #normalizing for Em=1
    ret[3] = c + Gov- w_e * l # resources
    ret[4] = phi*Uc*w_e
    ret[5] = rho2   
    ret[6] = Theta - 0.*Theta_ -Eps
    
    ret[7] = T # normalize average transfers to zero note doesn't depend on Eps here
    ret[8] = rho1_
    
    return ret
    
def f(y):
    '''
    Expectational equations that define e=Ef(y)
    '''
    logm,muhat,e,omega,I,c,l,rho1_,rho2,phi,w_e,UcP,a,x_,kappa_ = y
    
    m = np.exp(logm)
    mu = muhat * m    
    ret = np.empty(ne,dtype=y.dtype)
    ret[0] = UcP
    ret[1] = UcP * mu
    ret[2] = x_
    ret[3] = rho1_
    
    return ret
    
def Finv(YSS,z):
    '''
    Given steady state YSS solves for y_i
    '''
    logm,muhat,e,omega,I = z
    Theta,alpha1,alpha2,tau,eta,lamb,T = YSS
    
    m = np.exp(logm)
    mu = muhat * m
    
    Uc =alpha1/m
    c = Uc**(-1/sigma)
    Ucc =-sigma*c**(-sigma-1)  
    Ul = -(1-tau)*Uc*np.exp(e)
    l = (-Ul)**(1/gamma)
    Ull = -gamma*(l**(gamma-1))
    
    phi = ( omega*Ul - mu*(Ull*l + Ul) + lamb*np.exp(e) )/Ull
    
    rho1 = (omega*Uc - mu*(Ucc*c + Uc)- phi*np.exp(e)*(1-tau)*Ucc - lamb)/ (m*Ucc*(1-1./beta))
    rho2 = -rho1
    
    x = beta*( Uc*c + Ul*l )/(1-beta)
    
    kappa = rho1*Uc
    
    w_e = np.exp(e)    
    a = x/Uc
    
    return np.vstack((
    logm,muhat,e,omega,I,c,l,rho1,rho2,phi,w_e,Uc,a,x,kappa
    ))
    
def GSS(YSS,y_i,weights):
    '''
    Aggregate conditions for the steady state
    '''
    logm,muhat,e,omega,I,c,l,rho1,rho2,phi,w_e,UcP,a,x,kappa = y_i
    Theta,alpha1,alpha2,tau,eta,lamb,T = YSS
  
    
    Uc = c**(-sigma)
    
    return np.hstack((
    alpha1-alpha2, weights.dot(c+Gov-l*w_e),eta,weights.dot(phi*Uc*w_e),weights.dot(rho1),T,Theta  
    ))
    
def check_SS(YSS):
    '''
    Checks wether a solution to ss is valid
    '''
    #if (YSS[3] > 1.) | (YSS[3] < -.1):
    if (YSS[3] > 1.):    
        print 'negative of >100 percent taxes'
        return False
    else:    
        return True
    
def nomalize(Gamma):
    '''
    Normalizes the distriubtion of states if need be
    '''
                
    Gamma[:,0] -= np.mean(Gamma[:,0])
    Gamma[:,1] -= np.mean(Gamma[:,1])#/np.exp(Gamma[:,0]))*np.exp(Gamma[:,0])
    return Gamma
    
    
def check_extreme(z_i):
    return False