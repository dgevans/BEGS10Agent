import numpy as np
import scipy.optimize


thetas = np.array([330.,464.,695.,1070.,1602.])
thetas /= thetas[0]
Eps=np.hstack((np.ones(4)*0))
T=len(Eps)
N=len(thetas)
beta=0.98
psi=0.7
gov=0.22

def non_stochastic_economy(zz):
    l=zz[0:N]
    if min(l)>0  and max(l)<1:
        output=sum(l*thetas)
        tau=1-((1-psi)/psi)*(output-N*gov)/(sum(thetas)-output)
        c=(1-tau)*thetas*(1-l)*psi/(1-psi)
        b=((1-tau)*thetas*l-c)/(1-beta**-1)    
        # moments
        debt_gdp=(-sum(b)/N)/(output/N)
        transfers_gdp=(c[0]-(1-tau)*thetas[0]*l[0])/(output/N)
        fe=1./l-1
        avg_fe=fe.mean()    
        return tau,debt_gdp,transfers_gdp,avg_fe,max(fe)/min(fe)
    else:
        return (abs(l)-1)*10+100
        



target=0.2,0.75,.10,0.5,0.1

def calibrate(zz):
    return np.hstack((abs(np.array(non_stochastic_economy(zz))-np.array(target))))

    


res=scipy.optimize.root(calibrate,np.ones(5)*0.9)


print non_stochastic_economy(res.x),N*gov/sum(res.x*thetas)
        