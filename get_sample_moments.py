
import sys
sys.path.append('/usr/local/lib/python2.7/dist-packages')

bgp_flag=0
psi=0.5
if bgp_flag==1:
    import calibrate_begs_id_nu_bgp as Para
else:    
    import calibrate_begs_id_nu_ces as Para

Para.bgp_flag=bgp_flag

from mpi4py import MPI
from functools import partial
import time
import simulate_noMPI as simulate
import approximate_aggstate_noMPI as approximate
import numpy as np
import cPickle as pickle


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
    return np.corrcoef(np.array(zip(tfp,intrates)).T)



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
    
   

def run_simulation(ineq_slope,Gamma0,Eps,tfp_level,chi):
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
    return Gamma,Z,Y,Shocks,y,ineq_slope 
    
    #return data_simulation
T=100
max_num_sim=1
valEps = np.random.randn(T)
if Para.bgp_flag==1:
    data_calibration=pickle.load(open('data_calibration.pickle'))    
    valGamma0,psi,gov,thetas=data_calibration 
else:
    data_calibration=pickle.load(open('data_calibration_ces.pickle'))
    valGamma0,gamma,sigma,gov,thetas=data_calibration
    
Para.beta=Para.beta
Para.psi=psi
Para.sigma=sigma
Para.gamma=gamma
Para.Gov=gov
Para.chi=-0.06
Para.ineq_slope=1.5/.8  
tfp_level=1.
N=len(thetas)
chi=Para.chi
Para.sigma_E=0.04  
N=len(thetas)



start = time.clock()
X=np.hstack((np.ones(max_num_sim)*Para.ineq_slope,np.zeros(max_num_sim)))
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
s = comm.Get_size() #gets the number of processors
nX = len(X)/s
r = len(X)%s
my_range = slice(nX*rank+min(rank,r),nX*(rank+1)+min(rank+1,r))
run_simulation_partial = partial(run_simulation, Gamma0=valGamma0,Eps=valEps,tfp_level=1.0,chi=Para.chi)
my_data=map(run_simulation_partial,X[my_range]) 
data = comm.gather(my_data)
print '---Time taken----'
print time.clock()-start




returns,transfers,taxes,tfp,ineq_slope_index,var_covar,moments_std,moments_shocks_responses,moments_auto_corr={},{},{},{},{},{},{},{},{}

for i in range(max_num_sim*2):
    Gamma,Z,Y,Shocks,y,ineq_slope=data[0][i]
    transfers[i]=map(lambda t:Transfers(y,Y,t),range(1,T-2))
    returns[i]=map(lambda t: Returns(y,t), range(1,T-2))
    tfp[i]=map(lambda t: Y[t][Para.indx_Y['Theta']],range(1,T-2))
    taxes[i]=map(lambda t: Y[t][Para.indx_Y['tau']],range(1,T-2))
    ineq_slope_index[i]=ineq_slope
    moments_std[i]=[np.std(taxes[i]), np.std(transfers[i]),np.std(returns[i]),np.std(tfp[i])]
    moments_shocks_responses[i]= np.corrcoef((taxes[i],transfers[i],returns[i],tfp[i]))[3,:]
    moments_auto_corr[i]=[np.corrcoef(taxes[i][:-1], taxes[i][1:])[0,1],np.corrcoef(transfers[i][:-1],transfers[i][1:])[0,1]]


data_sample_moments=data,moments_std,moments_shocks_responses,moments_auto_corr
#with open('data_sample_moments.pickle', 'wb') as f:
#    pickle.dump(data_sample_moments, f)


