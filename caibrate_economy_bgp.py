import numpy as np
import scipy.optimize
import calibrate_begs_id_nu_bgp as Para
import simulate_noMPI as simulate
import approximate_aggstate_noMPI as approximate
import numpy as np
import cPickle as pickle




thetas = np.array([330.,464.,695.,1070.,1602.])

thetas /= thetas[0]
N=len(thetas)
beta=0.98


def non_stochastic_economy(zz):
    l=zz[0:N]
    psi=zz[N]
    gov=zz[N+1]    
    if min(l)>0  and max(l)<1:
        output=sum(l*thetas)
        tau=1-((1-psi)/psi)*(output-N*gov)/(sum(thetas)-output)
        c=(1-tau)*thetas*(1-l)*psi/(1-psi)
        b=((1-tau)*thetas*l-c)/(1-(1/beta))    
        atild = b[1:] - b[0]
        debt=sum(atild)/N
        debt_gdp=debt/(output/N)
        transfers_gdp=(c[0]-(1-tau)*thetas[0]*l[0])/(output/N)
        fe=1./l-1
        weights=l*thetas
        weights=weights/sum(weights)
        avg_fe=np.sum(fe*weights)
        gov_gdp=gov/(output/N)
        return tau,debt_gdp,transfers_gdp,avg_fe,gov_gdp
    else:
        return (abs(l[0:N])-1)*20+100
        

def get_allocation_non_stochastic_economy(zz):
    l=zz[0:N]
    psi=zz[N]
    gov=zz[N+1]    
    if min(l)>0  and max(l)<1:
        output=sum(l*thetas)
        tau=1-((1-psi)/psi)*(output-N*gov)/(sum(thetas)-output)
        c=(1-tau)*thetas*(1-l)*psi/(1-psi)
        b=((1-tau)*thetas*l-c)/(1-(1/beta))    
        atild = b[1:] - b[0]
        debt=sum(atild)/N
        debt_gdp=debt/(output/N)
        transfers_gdp=(c[0]-(1-tau)*thetas[0]*l[0])/(output/N)
        fe=1./l-1
        weights=l*thetas
        weights=weights/sum(weights)
        avg_fe=np.sum(fe*weights)
        gov_gdp=gov/(output/N)
        return c,l,tau,psi,gov,b
    else:
        return (abs(l[0:N])-1)*20+100

def calibrate(zz):
    weights=np.array([1000,100,1,10,100])
    return sum((weights*np.array(non_stochastic_economy(zz))-weights*np.array(target))**2)**0.5



def res_initial_conditions(zz,c,l,psi,tau):          
    mu=zz[0:N]
    lamb=zz[N]
    log_omega_N=zz[N+1]    
    omega=np.logspace(1,np.exp(log_omega_N),N)
    omega=omega/sum(omega)        
    Uc=psi/c
    logm=np.log(1./Uc)
    logm=logm-logm.mean()
    m=np.exp(logm)
    muhat=mu/m
    Ul=-(1-psi)/(1-l)
    Ull = -(1-psi)/((1-l)**2)
    Ucc =-psi/(c**2)
    e = np.log(thetas)    
    phi = ( omega*Ul - mu*(Ull*l + Ul) + lamb*np.exp(e) )/Ull
    rho1 = (omega*Uc - mu*(Ucc*c + Uc)- phi*np.exp(e)*(1-tau)*Ucc - lamb)/ (m*Ucc*(1-1./beta))
    ret=np.zeros(N+2)
    ret[0:N] = rho1
    ret[N]=sum(muhat)
    ret[N+1]=sum(phi*Uc*thetas)
    return ret
    
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
    
def t_gdp(y,Y,t):
    T=(y[t][0,Para.indx_y['c']]-y[t][0,Para.indx_y['w_e']]*y[t][0,Para.indx_y['l']]*(1-Y[t][Para.indx_Y['tau']]))

    return T/output(y[t])

def output(y):
    return sum(y[:,Para.indx_y['l']]*y[:,Para.indx_y['w_e']])/N

def debt_gdp(y):
    a = y[:,Para.indx_y['a']]
    atild = a[1:] - a[0]
    debt=sum(atild)/N
    

    return debt/output(y)



#-----------------#----------------------------



target=0.25,0.6,.1,0.5,0.2 #tax,debt_gdp,transfers_gdp,avg fe, gov_gdp ratio
    


# get allocation
res=scipy.optimize.minimize(calibrate,np.ones(7)*0.4)
print 'Targets attained:'
print non_stochastic_economy(res.x)
c,l,tau,psi,gov,b=get_allocation_non_stochastic_economy(res.x)


# get initial conditions mus 
zz=np.zeros(7)*.5
res2=scipy.optimize.root(res_initial_conditions,zz,args=(c,l,psi,tau))
mu=res2.x[0:N]
lamb=res2.x[N]
log_omega_N=res2.x[N+1]    
omega=np.logspace(1,np.exp(log_omega_N),N)
omega=omega/sum(omega)
logm=np.log(c)
market_weights=logm-logm.mean()
muhat=mu/np.exp(market_weights)


# check with simulation
Para.psi=psi
Para.beta=beta
Para.Gov=gov
Para.sigma_E=0   
Gamma={}
Gamma[0] = np.zeros((N,5))
Gamma[0][:,2] = np.log(thetas)
Gamma[0][:,3] = omega
Gamma[0][:,0] = market_weights
Gamma[0][:,1] = muhat
Gamma[0][:,4]=[.1,.25,.5,.75,.9]
Eps=np.zeros(4)
Gamma,Z,Y_tfp_ineq,Shocks,y_tfp_ineq =run_simulation(Gamma[0],Eps,ineq_slope=0.,tfp_level=1.,chi=0)    
print Y_tfp_ineq[1][Para.indx_Y['tau']],debt_gdp(y_tfp_ineq[1]),t_gdp(y_tfp_ineq,Y_tfp_ineq,1),np.mean(1/y_tfp_ineq[1][:,Para.indx_y['l']]-1),Para.Gov/output(y_tfp_ineq[1])

data_calibration=Gamma[0],psi,gov,thetas
with open('data_calibration.pickle', 'wb') as f:
    pickle.dump(data_calibration, f)
    