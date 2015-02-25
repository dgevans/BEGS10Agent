""" This script computes the intial conditions for the time 1 problem and pareto weights 
such that we meet targets for tau, T/GDP, debt/GDP and realtive assets from US data using deterministic economy"""

import numpy as np
import scipy.optimize
import calibrate_begs_id_nu_ces as Para
import cPickle as pickle

""" Momenta from wage, government budget, asset data"""
thetas = np.array([330.,464.,695.,1070.,1602.]) # from cps
gamma=2.0
sigma=1.0
Para.sigma=sigma
Para.gamma=gamma
thetas /= thetas[0]
N=len(thetas)

debt_y=0.6
transfers_gdp=0.1
gov_gdp=0.12
tau=debt_y*((Para.beta)/(1-Para.beta))**-1+transfers_gdp+gov_gdp # no shock budget constraint
      
b9050=649./72
b7550=172./72
b5025=19./72
b2=np.array([b5025,1,b7550,b9050]) # from SCF with b_1 normalized to zero.





def res_initial_conditions(zz,c,l,tau):          
    """ Given an allocation, this function backs out multipliers and pareto weights
    such that this allocation is a complete markets continuation ramsey allocation 
    for an economy with no shocks"""  
    mu=zz[0:N]
    lamb=zz[N]
    log_omega_N=zz[N+1]    
    omega=np.logspace(1,np.exp(log_omega_N),N)
    omega=omega/sum(omega)        
    Uc = c**(-Para.sigma)
    Ucc = -sigma*c**(-Para.sigma-1)
    Ul = -l**(Para.gamma)
    Ull = -Para.gamma*l**(Para.gamma-1)
    logm=np.log(1./Uc)
    logm=logm-logm.mean()
    m=np.exp(logm)
    muhat=mu/m
    e = np.log(thetas)    
    phi = ( omega*Ul - mu*(Ull*l + Ul) + lamb*np.exp(e) )/Ull
    rho1 = (omega*Uc - mu*(Ucc*c + Uc)- phi*np.exp(e)*(1-tau)*Ucc - lamb)/ (m*Ucc*(1-1./Para.beta))
    ret=np.zeros(N+2)
    ret[0:N] = rho1
    ret[N]=sum(muhat)
    ret[N+1]=sum(phi*Uc*thetas)
    return ret



    
def labor_supply(tau,b_i,theta_i):
    """: This function solves for the implementability constraint of agent i for labor supply"""
    res=scipy.optimize.root(lambda l_i: ((1-tau)*theta_i*l_i-(((1-tau)*theta_i*(l_i**(-gamma)))**(1/sigma)))/(1-1/Para.beta)-b_i,0.5)
    return res.x

def error_calibrate(Y,b2):    
    """ This is an interim function to compute residuals. 
    : Using the targeted T-gdp ratio, tau and assets, we back out labor supply
    from the implementability constraint for each agent and the residual is obtained bfrom feasibility.
    """  
    T=Y*transfers_gdp
    b1=T*Para.beta/(1-Para.beta)
    b=np.hstack((b1,b2+b1))    
    l=np.ones(N)
    for i in range(N):
        l[i]=labor_supply(tau,b[i],thetas[i])
    return Y-sum(thetas*l)/N
    





# Get the allocation c,l,b that meets the targets for tax rate, transfers to gdp and relative assets 
res=scipy.optimize.root(error_calibrate,2,args=b2)
Y=res.x
gov=Y*gov_gdp
T=Y*transfers_gdp
b1=T*Para.beta/(1-Para.beta)
b=np.hstack((b1,b2+b1))
l=np.ones(N)
for i in range(N):
    l[i]=labor_supply(tau,b[i],thetas[i])
    
c=(((1-tau)*thetas*(l**(-gamma)))**(1/sigma))
b=((1-tau)*thetas*l-c)/(1-1/Para.beta) 
Y=sum(thetas*l)/N
T=c[0]-(1-tau)*thetas[0]*l[0]
transfers_gdp=T/Y
atild=b[1:]-b[0]
debt_y=(sum(atild)/N)/(Y)
gov=(sum(thetas*l)-sum(c))/N
gov_gdp=gov/Y

print tau,debt_y,transfers_gdp,gov_gdp
 


# Obtain the OMEGAs and initial multipliers for the time 1 continuation problem that supports this allocation
zz=np.zeros(N+2)*.5
res2=scipy.optimize.root(res_initial_conditions,zz,args=(c,l,tau))
mu=res2.x[0:N]
lamb=res2.x[N]
log_omega_N=res2.x[N+1]    
omega=np.logspace(1,np.exp(log_omega_N),N)
omega=omega/sum(omega)
logm=np.log(c)
market_weights=logm-logm.mean()
muhat=mu/np.exp(market_weights)



# Store the calbration
Gamma={}
Gamma[0] = np.zeros((N,5))
Gamma[0][:,2] = np.log(thetas)
Gamma[0][:,3] = omega
Gamma[0][:,0] = market_weights
Gamma[0][:,1] = muhat
Gamma[0][:,4]=[.1,.25,.5,.75,.9]

data_calibration=Gamma[0],gamma,sigma,gov,thetas
with open('data_calibration_ces.pickle', 'wb') as f:
    pickle.dump(data_calibration, f)


# This code checks with simulation code  
#==============================================================================
# def run_simulation(Gamma0,Eps,ineq_slope=2/.9,tfp_level=1.,chi=0):
#     Gamma,Z,Y,Shocks,y = {},{},{},{},{}
#     Para.ineq_slope=ineq_slope
#     Para.chi=chi
#     Para.tfp_level=tfp_level
#     approximate.calibrate(Para)
#     simulate.approximate = approximate
#     T=len(Eps)
#     Gamma[0] = Gamma0
#     Z[0] = 0.
#     simulate.simulate_aggstate(Para,Gamma,Z,Y,Shocks,y,T-1,agg_shocks= Eps)    
#     return Gamma,Z,Y,Shocks,y 
#   
# def t_gdp(y,Y,t):
#     T=(y[t][0,Para.indx_y['c']]-y[t][0,Para.indx_y['w_e']]*y[t][0,Para.indx_y['l']]*(1-Y[t][Para.indx_Y['tau']]))
#     return T/output(y[t])
# 
# def output(y):
#     return sum(y[:,Para.indx_y['l']]*y[:,Para.indx_y['w_e']])/N
# 
# def debt_gdp(y):
#     a = y[:,Para.indx_y['a']]
#     atild = a[1:] - a[0]
#     debt=sum(atild)/N
#     return debt/output(y)
# Para.gamma=gamma
# Para.sigma=sigma
# Para.Gov=gov
# Para.sigma_E=0   
# Eps=np.zeros(4)
# Gamma,Z,Y_tfp_ineq,Shocks,y_tfp_ineq =run_simulation(Gamma[0],Eps,ineq_slope=0.,tfp_level=1.,chi=0)    
# print Y_tfp_ineq[1][Para.indx_Y['tau']],debt_gdp(y_tfp_ineq[1]),t_gdp(y_tfp_ineq,Y_tfp_ineq,1),Para.Gov/output(y_tfp_ineq[1])
#==============================================================================

       