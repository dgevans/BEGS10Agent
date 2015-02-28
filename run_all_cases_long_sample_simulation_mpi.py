""" This file generates the 3000 period simulation for all cases 
1) with inequality shocks 
2) without inequlity shocks
3) actual shocks
4) conditional mean
"""

import sys
sys.path.append('/usr/local/lib/python2.7/dist-packages')
bgp_flag=0
if bgp_flag==1:
    import calibrate_begs_id_nu_bgp as Para
else:    
    import calibrate_begs_id_nu_ces as Para

Para.bgp_flag=bgp_flag

import time
import simulate_noMPI as simulate
from mpi4py import MPI
import approximate_aggstate_noMPI as approximate
import numpy as np
import cPickle as pickle
from functools import partial


    

def run_simulation(ex,Gamma0,Eps): ## function to run all cases with MPI and store the simulation
    Para.chi=Case[ex][0]
    conditional_mean_flag=Case[ex][1]
    inequlity_shock_flag=Case[ex][2]
    Gamma,Z,Y,Shocks,y = {},{},{},{},{}
    T=len(Eps)
    Gamma[0] = Gamma0
    Z[0] = 0.
   
    if inequlity_shock_flag==0:
       Para.ineq_slope=0  
    else:
       Para.ineq_slope= 1.0/0.8
        
    approximate.calibrate(Para)
    simulate.approximate = approximate
    
    
    if conditional_mean_flag==1:    
        simulate.simulate_aggstate_ConditionalMean(Para,Gamma,Z,Y,Shocks,y,T-1,agg_shocks= Eps)
    else:
        simulate.simulate_aggstate(Para,Gamma,Z,Y,Shocks,y,T-1,agg_shocks= Eps)    
            
    
    data_simulation=Gamma,Z,Y,Shocks,y,Case[ex] 
    with open('data_simulation_new'+str(ex)+'.pickle', 'wb') as f:
        pickle.dump(data_simulation, f)
    
############################################################################################
    

# get the appropiate cali bration and initial conditions
Gamma={}
if Para.bgp_flag==1:
    data_calibration=pickle.load(open('data_calibration.pickle'))    
    Gamma0,psi,gov,thetas=data_calibration 
    Para.psi=psi
    Para.Gov=gov

else:
    data_calibration=pickle.load(open('data_calibration_ces.pickle'))
    Gamma0,gamma,sigma,gov,thetas=data_calibration
    Para.sigma=sigma
    Para.gamma=gamma
    Para.Gov=gov


Gamma[0]=Gamma0
N=len(thetas)
T=5000   

# List all the cases
chi_grid=np.array([-1.50,-1.0,-.5,-0.06,.5,1.0,1.5])
conditional_mean_flag=np.array([0])
ineq_shocks_flag=np.array([1])
Case={}
k=0
for j in range(len(conditional_mean_flag)):
    for h in range(len(ineq_shocks_flag)):
        for i in range(len(chi_grid)):
            Case[k]=[chi_grid[i],conditional_mean_flag[j],ineq_shocks_flag[h]]
            k=k+1 

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
# common shocks, store them
if rank ==0:
    Eps = np.random.randn(T)
    Eps=np.minimum(3.,np.maximum(-3.,Eps))
    with open('Eps.pickle', 'wb') as f:
               pickle.dump(Eps, f)
else:
    Eps = None

Eps = comm.bcast(Eps) #ALl using same Shocks    

   
   
# run all cases using MPI   
start = time.clock()
X=range(len(Case))

s = comm.Get_size() #gets the number of processors
nX = len(X)/s
r = len(X)%s
my_range = slice(nX*rank+min(rank,r),nX*(rank+1)+min(rank+1,r))
run_simulation_partial = partial(run_simulation, Gamma0=Gamma[0],Eps=Eps)
print 'Xrange= '  
print X[my_range]
map(run_simulation_partial,X[my_range]) 


print '---Time taken----'
print time.clock()-start


