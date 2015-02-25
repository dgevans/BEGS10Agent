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
T=200
debt=[]
alphalist=[]

for alpha in np.linspace(.03,.16,15):
    Gamma,Z,Y,Shocks,y = {},{},{},{},{}
    Gamma[0]=Gamma0
    Gamma[0][:,3] = np.linspace(0.2-alpha,0.2+alpha,N)    
    Z[0] = 0.
    Para.sigma_E = 0.25#allows for quicker convergence
    approximate.calibrate(Para)
    simulate.approximate = approximate
    Eps = np.random.randn(T)
    Eps=np.minimum(3.,np.maximum(-3.,Eps))
    simulate.simulate_aggstate_ConditionalMean(Para,Gamma,Z,Y,Shocks,y,T-1,agg_shocks= Eps)
    Para.sigma_E = 0.025
    approximate.calibrate(Para)
    approx = approximate.approximate(Gamma[T-2])    
    data = {}
    data=approx.iterate(0.)[4]
    debt.append(-GovernmentDebt(data))
    alphalist.append(alpha)
    
plt.rc('text', usetex=True)
f,(ax1) =plt.subplots(1,1)
lines_taxes=ax1.plot(alphalist[:-1],debt[:-1])

ax1.set_xlabel(r'$\mathbf{\alpha}$')
ax1.set_ylabel(r'\text{Govt. Assets}')
# tfp+ineq bold
plt.setp(lines_taxes[0],color='k',linewidth=2)

plt.savefig('comp_stats_alpha.pdf',dpi=300)        
    