bgp_flag=0
psi=0.5
if bgp_flag==1:
    import calibrate_begs_id_nu_bgp as Para
else:    
    import calibrate_begs_id_nu_ces as Para

Para.bgp_flag=bgp_flag

import simulate_noMPI as simulate
import approximate_aggstate_noMPI as approximate
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt


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
    

def compute_cov_int_output(y,T):
    
    intrates=map(lambda t: Returns(y,t),range(1,T-2))
    output=map(lambda t: sum(y[t][:,Para.indx_y['l']]*y[t][:,Para.indx_y['w_e']]),range(1,T-2))
    return np.cov(np.array(zip(output,intrates)).T)


def compute_cov_int_eps(y,Y,T):
    
    intrates=map(lambda t: Returns(y,t),range(1,T-2))
    tfp=map(lambda t: Y[t][Para.indx_Y['Theta']],range(1,T-2))
    return np.cov(np.array(zip(tfp,intrates)).T)



def compute_cov_output_eps(y,Y,T):
    
    output=map(lambda t: sum(y[t][:,Para.indx_y['l']]*y[t][:,Para.indx_y['w_e']]),range(1,T-2))
    tfp=map(lambda t: Y[t][Para.indx_Y['Theta']],range(1,T-2))
    return np.cov(np.array(zip(tfp,output)).T)




def  consumption_ineq(y):
    # consumption
    consumption_agent_1=y[0,Para.indx_y['c']]
    consumption_agent_N=y[N-1,Para.indx_y['c']]
    c_ineq=consumption_agent_N/consumption_agent_1
    return c_ineq
    
def  earnings_ineq(y):
    # earnings
    earnings_agent_1=y[0,Para.indx_y['l']]*y[0,Para.indx_y['w_e']]
    earnings_agent_N=y[N-1,Para.indx_y['l']]*y[N-1,Para.indx_y['w_e']]
    e_ineq=earnings_agent_N/earnings_agent_1
    return e_ineq
    

def  labor_ineq(y):
    # labor
    labor_agent_1=y[0,Para.indx_y['l']]
    labor_agent_N=y[N-1,Para.indx_y['l']]
    l_ineq=labor_agent_N/labor_agent_1
    return l_ineq

    

def  assets_ineq(y):
    # assets
    assets_agent_N=y[N-1,Para.indx_y['a']]-y[0,Para.indx_y['a']]
    total_assets=N*GovernmentDebt(y) 
    return assets_agent_N/total_assets
    
    

def get_chi(y,Y,t):
    UcP=y[t][1,Para.indx_y['UcP']]
    Uc= Mu(y[t][1,Para.indx_y['c']] )
    P=UcP/Uc
    Theta=Y[t][Para.indx_Y['Theta']]
    eps=Theta
    chi=(P-1)/eps
    return chi
    

def Transfers(y,Y,t):
    T=(y[t][0,Para.indx_y['c']]-y[t][0,Para.indx_y['w_e']]*y[t][0,Para.indx_y['l']]*(1-Y[t][Para.indx_Y['tau']]))

    return T


def GovernmentDebt(y):
    a = y[:,Para.indx_y['a']]
    atild = a[1:] - a[0]
    debt=sum(atild)/N

    return debt
    
   

def run_simulation(Gamma0,Eps,ineq_slope=2/.9,tfp_level=1.,chi=0):
    Gamma,Z,Y,Shocks,y = {},{},{},{},{}
    Para.ineq_slope=ineq_slope
    Para.chi=chi
    Para.tfp_level=tfp_level
    approximate.calibrate(Para)
    simulate.approximate = approximate
    T=len(Eps)
    Gamma[0] = Gamma0
    Z[0] = 0.
    simulate.simulate_aggstate(Para,Gamma,Z,Y,Shocks,y,T-1,agg_shocks= Eps)    
    return Gamma,Z,Y,Shocks,y 
    
Eps=np.hstack((np.ones(5)*0.,-1*np.ones(4),np.ones(5)*0))
Eps_only_bad_shocks=np.hstack((np.ones(100)*-1.0))

T=len(Eps)
if Para.bgp_flag==1:
    data_calibration=pickle.load(open('data_calibration.pickle'))    
    Gamma0,psi,gov,thetas=data_calibration 
else:
    data_calibration=pickle.load(open('data_calibration_ces.pickle'))
    Gamma0,gamma,sigma,gov,thetas=data_calibration
    
Para.beta=Para.beta
Para.psi=psi
Para.sigma=sigma
Para.gamma=gamma
Para.Gov=gov
Para.chi=-0.06
  
N=len(thetas)

Gamma={}
Gamma[0]=Gamma0
Gamma,Z,Y_tfp_ineq,Shocks,y_tfp_ineq =run_simulation(Gamma[0],Eps,ineq_slope=2./.8,tfp_level=1.,chi=Para.chi)
Gamma,Z,Y_only_ineq,Shocks,y_only_ineq =run_simulation(Gamma[0],Eps,ineq_slope=2./.8,tfp_level=0.,chi=Para.chi)
Gamma,Z,Y_only_tfp,Shocks,y_only_tfp =run_simulation(Gamma[0],Eps,ineq_slope=0.,tfp_level=1.,chi=Para.chi)

f,(ax1,ax2,ax3) =plt.subplots(3,1,sharex='col')
lines_taxes=ax1.plot(np.vstack(np.array((map(lambda t: Y_tfp_ineq[t][3],range(1,T-2)),map(lambda t: Y_only_tfp[t][3],range(1,T-2)),map(lambda t: Y_only_ineq[t][3],range(1,T-2))))).T)
lines_debt=ax2.plot(np.vstack((map(lambda t: GovernmentDebt(y_tfp_ineq[t]),range(1,T-2)),map(lambda t: GovernmentDebt(y_only_tfp[t]),range(1,T-2)),map(lambda t: GovernmentDebt(y_only_ineq[t]),range(1,T-2)))).T)
lines_transfers=ax3.plot(np.vstack(np.array((map(lambda t: Transfers(y_tfp_ineq,Y_tfp_ineq,t),range(1,T-2)),map(lambda t: Transfers(y_only_tfp,Y_only_tfp,t),range(1,T-2)),map(lambda t: Transfers(y_only_ineq,Y_only_ineq,t),range(1,T-2))))).T)

ax1.set_title(r'Taxes')
ax2.set_title(r'Debt')
ax3.set_title(r'Transfers')

# tfp+ineq bold
plt.setp(lines_taxes[0],color='k',linewidth=2)
plt.setp(lines_debt[0],color='k',linewidth=2)
plt.setp(lines_transfers[0],color='k',linewidth=2)

# only tfp --
plt.setp(lines_taxes[1],color='k',linewidth=2,ls='--')
plt.setp(lines_transfers[1],color='k',linewidth=2,ls='--')
plt.setp(lines_debt[1],color='k',linewidth=2,ls='--')

# only ineq :
plt.setp(lines_taxes[2],color='k',linewidth=2,ls=':')
plt.setp(lines_transfers[2],color='k',linewidth=2,ls=':')
plt.setp(lines_debt[2],color='k',linewidth=2,ls=':')

ax1.axvspan(7, 3, facecolor='k', alpha=0.25)
ax2.axvspan(7, 3, facecolor='k', alpha=0.25)
ax3.axvspan(7, 3, facecolor='k', alpha=0.25)

ax1.locator_params(nbins=4)
ax2.locator_params(nbins=4)
ax3.locator_params(nbins=5)
ax1.autoscale(tight=True,axis=u'both')
ax2.autoscale(tight=True,axis=u'both')
ax3.autoscale(tight=True,axis=u'both')

plt.xlabel('t')    

plt.tight_layout()

plt.savefig('irf_bm_chi_shocks.png',dpi=300)        
print map(lambda t: Returns(y_only_tfp,t), range(1,10))



Eps_only_bad_shocks=np.hstack((np.ones(100)*-1.0))

T=len(Eps_only_bad_shocks)
    

Gamma={}
Gamma[0]=Gamma0
Gamma,Z,Y_only_bad_shocks,Shocks,y_only_bad_shocks =run_simulation(Gamma[0],Eps_only_bad_shocks,ineq_slope=2./.8,tfp_level=1.,chi=Para.chi)

f,(ax1) =plt.subplots(1,1)
lines_taxes=ax1.plot(np.vstack(np.array((map(lambda t: Y_only_bad_shocks[t][3],range(1,T-2)),map(lambda t: Y_only_bad_shocks[t][3],range(1,T-2)),map(lambda t: Y_only_bad_shocks[t][3],range(1,T-2))))).T)

ax1.set_title(r'Taxes')

# tfp+ineq bold
plt.setp(lines_taxes,color='k',linewidth=2)
plt.xlabel('t')    

plt.tight_layout()

plt.savefig('taxes_only_bad_shocks.png',dpi=300)        






