""" This file generates the plots the raw simulations. We do the benchmark and two cases with low and high chi
"""
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
import sys
sys.path.append('/usr/local/lib/python2.7/dist-packages')
bgp_flag=0
if bgp_flag==1:
    import calibrate_begs_id_nu_bgp as Para
else:    
    import calibrate_begs_id_nu_ces as Para

Para.bgp_flag=bgp_flag

if Para.bgp_flag==1:
    data_calibration=pickle.load(open('data_calibration.pickle'))    
    Gamma0,psi,gov,thetas=data_calibration 
    Para.psi=psi
else:
    data_calibration=pickle.load(open('data_calibration_ces.pickle'))
    Gamma0,gamma,sigma,gov,thetas=data_calibration
    Para.sigma=sigma
    Para.gamma=gamma

#choose the cases that need to be plotted
Para.Gov=gov
Eps= pickle.load(open('Eps.pickle'))
data_simulation_b= pickle.load(open('data_simulation_new3.pickle'))
data_simulation0= pickle.load(open('data_simulation_new0.pickle'))
data_simulation1= pickle.load(open('data_simulation_new6.pickle'))

Gamma2,Z2,Y2,Shocks2,y2=data_simulation0
Gamma,Z,Y,Shocks,y=data_simulation_b
Gamma3,Z3,Y3,Shocks3,y3=data_simulation1

T=len(Gamma)
N=len(y[0])
T=T-500


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

def Mu(c):
    if Para.bgp_flag==0:
       return c**(-Para.sigma)
    else:
       return Para.psi/c  
    

def Returns(y,t):
    Uc_ = Mu(y[t-1][1,Para.indx_y['c']])
    UcP=y[t][1,Para.indx_y['UcP']]
    Uc= Mu(y[t][1,Para.indx_y['c']]) 
    P=UcP/Uc
    kappa_=y[t][1,Para.indx_y['kappa_']]
    rho1_=y[t][1,Para.indx_y['rho1_']]
    EUcp=kappa_/rho1_
    q=Para.beta*EUcp/Uc_
    return P/q


def get_chi(y,Y,t):
    UcP=y[t][1,Para.indx_y['UcP']]
    Uc= Mu(y[t][1,Para.indx_y['c']])
    P=UcP/Uc
    Theta=Y[t][Para.indx_Y['Theta']]
    eps=Theta
    chi=(P-1)/eps
    return chi
    
    

f,(ax1,ax2,ax3) =plt.subplots(3,1,sharex='col')
lines_taxes=ax2.plot(range(0,T-2,4),np.vstack(np.array((map(lambda t: Y[t][3],range(0,T-2,4)),map(lambda t: Y2[t][3],range(0,T-2,4)),map(lambda t: Y3[t][3],range(0,T-2,4))))).T)
lines_debt=ax1.plot(range(0,T-2,4),np.vstack((map(lambda t: GovernmentDebt(y[t]),range(0,T-2,4)),map(lambda t: GovernmentDebt(y2[t]),range(0,T-2,4)),map(lambda t: GovernmentDebt(y3[t]),range(0,T-2,4)))).T)
lines_transfers=ax3.plot(range(0,T-2,4),np.vstack(np.array((map(lambda t: Transfers(y,Y,t),range(0,T-2,4)),map(lambda t: Transfers(y2,Y,t),range(0,T-2,4)),map(lambda t: Transfers(y3,Y,t),range(0,T-2,4))))).T)
ax2.set_title(r'taxes-rates')
ax1.set_title(r'debt-gdp ratio')
ax3.set_title(r'transfers-gdp')

plt.setp(lines_taxes[0],color='k',linewidth=2,linestyle='-')
plt.setp(lines_debt[0],color='k',linewidth=2,linestyle='-')
plt.setp(lines_transfers[0],color='k',linewidth=2,linestyle='-')

plt.setp(lines_taxes[1],color='k',linewidth=2,linestyle='--')
plt.setp(lines_debt[1],color='k',linewidth=2,linestyle='--')
plt.setp(lines_transfers[1],color='k',linewidth=2,linestyle='--')

plt.setp(lines_taxes[2],color='k',linewidth=2,linestyle=':')
plt.setp(lines_debt[2],color='k',linewidth=2,linestyle=':')
plt.setp(lines_transfers[2],color='k',linewidth=2,linestyle=':')

plt.xlabel('t')    
plt.tight_layout()
ax1.locator_params(nbins=6)
ax2.locator_params(nbins=5)
ax3.locator_params(nbins=5)

plt.savefig('long_simulation_new_debt.png',dpi=300)        
