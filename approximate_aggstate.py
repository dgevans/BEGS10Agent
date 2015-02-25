# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 17:28:52 2014

@author: dgevans
"""
import steadystate
import numpy as np
import utilities
from utilities import hashable_array
from utilities import quadratic_dot
from utilities import dict_fun
import itertools
from scipy.cluster.vq import kmeans2
from scipy.optimize import root

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

timing = np.zeros(6) 

def parallel_map(f,X):
    '''
    A map function that applies f to each element of X
    '''
    s = comm.Get_size() #gets the number of processors
    nX = len(X)/s
    r = len(X)%s
    my_range = slice(nX*rank+min(rank,r),nX*(rank+1)+min(rank+1,r))
    my_data =  map(f,X[my_range])
    data = comm.gather(my_data)
    if rank == 0:
        return list(itertools.chain(*data))
    else:
        return None
        
def parallel_map_noret(f,X):
    '''
    A map function that applies f to each element of X
    '''
    s = comm.Get_size() #gets the number of processors
    nX = len(X)/s
    r = len(X)%s
    my_range = slice(nX*rank+min(rank,r),nX*(rank+1)+min(rank+1,r))
    map(f,X[my_range])
    
def parallel_sum(f,X):
    '''
    In parallel applies f to each element of X and computes the sum
    '''
    s = comm.Get_size() #gets the number of processors
    nX = len(X)/s
    r = len(X)%s
    my_range = slice(nX*rank+min(rank,r),nX*(rank+1)+min(rank+1,r))
    my_sum =  sum(itertools.imap(f,X[my_range]))
    if r==0:#if evenly split do it fast
        shape = my_sum.shape
        ret = np.empty(shape)
        comm.Allreduce([my_sum,MPI.DOUBLE],[ret,MPI.DOUBLE])
        return ret
    else:
        sums = comm.gather(my_sum)
        if rank == 0:
            res = sum(sums)
            comm.bcast(res.shape)
        else:
            shape = comm.bcast(None)
            res = np.empty(shape)
        comm.Bcast([res,MPI.DOUBLE])
        return res
    
def parallel_dict_map(F,l):
    '''
    perform a map preserving the dict structure
    '''
    ret = {}
    temp = parallel_map(F,l)
    keys = temp[0].keys()
    for key in keys:
        ret[key] = [t[key] for t in temp]
    return ret
    
    
F =None
G = None
f = None

n = None
nG = None
ny = None
ne = None
nY = None
nz = None
nv = None
n_p = None
nZ = None
neps = None
Ivy = None
Izy = None
IZY = None
Para = None

shock = None

logm_min = -np.inf
muhat_min = -np.inf

y,e,Y,z,v,eps,Eps,p,Z,S,sigma,sigma_E = [None]*12
#interpolate = utilities.interpolator_factory([3])

def calibrate(Parahat):
    global F,G,f,ny,ne,nY,nz,nv,n,Ivy,Izy,IZY,nG,n_p,nZ,neps
    global y,e,Y,z,v,eps,p,Z,S,sigma,Eps,sigma_E,Para
    Para = Parahat
    #global interpolate
    F,G,f = Para.F,Para.G,Para.f
    ny,ne,nY,nz,nv,n,nG,n_p,neps,nZ = Para.ny,Para.ne,Para.nY,Para.nz,Para.nv,Para.n,Para.nG,Para.n_p,Para.neps,Para.nZ
    Ivy,Izy,IZY = np.zeros((nv,ny)),np.zeros((nz,ny)),np.zeros((nZ,nY)) # Ivy given a vector of y_{t+1} -> v_t and Izy picks out the state
    Ivy[:,-nv:] = np.eye(nv)
    Izy[:,:nz] = np.eye(nz)
    IZY[:,:nZ] = np.eye(nZ)
        
    
    #store the indexes of the various types of variables
    y = np.arange(ny).view(hashable_array)
    e = np.arange(ny,ny+ne).view(hashable_array)
    Y = np.arange(ny+ne,ny+ne+nY).view(hashable_array)
    z = np.arange(ny+ne+nY,ny+ne+nY+nz).view(hashable_array)
    v = np.arange(ny+ne+nY+nz,ny+ne+nY+nz+nv).view(hashable_array)
    p = np.arange(ny+ne+nY+nz+nv,ny+ne+nY+nz+nv+n_p).view(hashable_array)
    Z = np.arange(ny+ne+nY+nz+nv+n_p,ny+ne+nY+nz+nv+n_p+nZ).view(hashable_array)
    eps = np.arange(ny+ne+nY+nz+nv+n_p+nZ,ny+ne+nY+nz+nv+n_p+nZ+neps).view(hashable_array)
    Eps = np.arange(ny+ne+nY+nz+nv+n_p+nZ+neps,ny+ne+nY+nz+nv+n_p+nZ+neps+1).view(hashable_array)
    
    S = np.hstack((z,Y)).view(hashable_array)
    
    sigma = Para.sigma_e.view(hashable_array)
    
    sigma_E = Para.sigma_E
    
    #interpolate = utilities.interpolator_factory([3]*nz) # cubic interpolation
    steadystate.calibrate(Para)

class approximate(object):
    '''
    Computes the second order approximation 
    '''
    def __init__(self,Gamma,fit = True):
        '''
        Approximate the equilibrium policies given z_i
        '''
        self.Gamma = Gamma
        self.approximate_Gamma()
        self.ss = steadystate.steadystate(self.dist)

        #precompute Jacobians and Hessians
        self.get_w = dict_fun(self.get_wf)
        self.DF = dict_fun(lambda z_i:utilities.ad_Jacobian(F,self.get_w(z_i)))
        self.HF = dict_fun(lambda z_i:utilities.ad_Hessian(F,self.get_w(z_i)))
        
        self.DG = dict_fun(lambda z_i:utilities.ad_Jacobian(G,self.get_w(z_i)))
        self.HG = dict_fun(lambda z_i:utilities.ad_Hessian(G,self.get_w(z_i)))
        
        self.df = dict_fun(lambda z_i:utilities.ad_Jacobian(f,self.get_w(z_i)[y]))
        self.Hf = dict_fun(lambda z_i:utilities.ad_Hessian(f,self.get_w(z_i)[y]))
        
        
        #linearize
        if fit:
            self.linearize()
            self.quadratic()
            self.join_function()
        
    def approximate_Gamma(self,k=304):
        '''
        Approximate the Gamma distribution
        '''
        #if rank == 0:
        #    cluster,labels = kmeans2(self.Gamma,k,minit='points')
        #    cluster,labels = comm.bcast((cluster,labels))
        #else:
        #    cluster,labels = comm.bcast(None)
        cluster,labels = kmeans2(self.Gamma,self.Gamma[:k,:],minit='matrix')
        weights = (labels-np.arange(k).reshape(-1,1) ==0).sum(1)/float(len(self.Gamma))
        #Para.nomalize(cluster,weights)
        self.Gamma_ss = cluster[labels,:]
        mask = weights > 0
        cluster = cluster[mask]
        weights = weights[mask]
        
        
        self.dist = zip(cluster,weights)        
        

        
    def integrate(self,f):
        '''
        Integrates a function f over Gamma
        '''
        def f_int(x):
            z,w = x
            return w*f(z)
        return parallel_sum(f_int,self.dist)
    
    def get_wf(self,z_i):
        '''
        Gets w for particular z_i
        '''
        ybar = self.ss.get_y(z_i).flatten()
        Ybar = self.ss.get_Y()
        Zbar = Ybar[:nZ]
        ebar = f(ybar)
        
        return np.hstack((
        ybar,ebar,Ybar,ybar[:nz],Ivy.dot(ybar),np.ones(n_p),Zbar,np.zeros(neps),0.
        ))
        
    def compute_dy(self):
        '''
        Computes dy w.r.t z_i, eps, Y
        '''
        self.dy = {}
        
        def dy_z(z_i):
            DF = self.DF(z_i)
            df = self.df(z_i)
            DFi = DF[n:] # pick out the relevant equations from F. Forexample to compute dy_{z_i} we need to drop the first equation
            return np.linalg.solve(DFi[:,y] + DFi[:,e].dot(df)+DFi[:,v].dot(Ivy),
                                    -DFi[:,z])
        self.dy[z] = dict_fun(dy_z)
        
        def dy_eps(z_i):   
            DF = self.DF(z_i)    
            DFi = DF[:-n,:]
            return np.linalg.solve(DFi[:,y] + DFi[:,v].dot(Ivy).dot(self.dy[z](z_i)).dot(Izy),
                                    -DFi[:,eps])
                                    
        self.dy[eps] = dict_fun(dy_eps)
        
        def dy_Eps(z_i):   
            DF = self.DF(z_i)    
            DFi = DF[:-n,:]
            return np.linalg.solve(DFi[:,y] + DFi[:,v].dot(Ivy).dot(self.dy[z](z_i)).dot(Izy),
                                    -DFi[:,Eps])
                                    
        self.dy[Eps] = dict_fun(dy_Eps)
        self.compute_dy_Z()
                
        def dy_Y(z_i):
            DF = self.DF(z_i)
            df = self.df(z_i)                    
            DFi = DF[n:,:]
            return np.linalg.solve(DFi[:,y] + DFi[:,e].dot(df) + DFi[:,v].dot(Ivy),
                                -DFi[:,Y] - DFi[:,v].dot(Ivy).dot(self.dy[Z](z_i)).dot(IZY))
        self.dy[Y] = dict_fun(dy_Y)
        
        def dy_YEps(z_i):   
            DF = self.DF(z_i)    
            DFi = DF[:-n,:]
            return np.linalg.solve(DFi[:,y] + DFi[:,v].dot(Ivy).dot(self.dy[z](z_i)).dot(Izy),
                                    -DFi[:,Y])
                                    
        self.dy[Y,Eps] = dict_fun(dy_YEps)
        
    def residual_test(self,dZ_Z):
        #right now assume nZ =1
        def dy_Z(z_i):
            DFi = self.DF(z_i)[n:,:]
            df = self.df(z_i)
            DFhat_Zinv = np.linalg.inv(DFi[:,y] + DFi[:,e].dot(df) + DFi[:,v].dot(Ivy)*dZ_Z[0]) 
            return DFhat_Zinv.dot(-DFi[:,Z]),DFhat_Zinv.dot(-DFi[:,Y])
            
        DG = lambda z_i : self.DG(z_i)[nG:,:]
            
        A,B = lambda z_i : dy_Z(z_i)[0], lambda z_i : dy_Z(z_i)[1]
        temp1 = self.integrate(  lambda z_i: DG(z_i)[:,y].dot(A(z_i)) + DG(z_i)[:,Z]  )
        temp2 = self.integrate(  lambda z_i: DG(z_i)[:,y].dot(B(z_i)) + DG(z_i)[:,Y]  )
        dY_Z  =  np.linalg.solve(temp2,-temp1)
            
        return dY_Z[:nZ]-dZ_Z
        
    def compute_dy_Z(self):
        '''
        Computes linearization w/ respecto aggregate state.
        '''
        self.dZ_Z = np.eye(nZ)
        
        #right now assume nZ =1
        def dy_Z(z_i):
            DFi = self.DF(z_i)[n:,:]
            df = self.df(z_i)
            DFhat_Zinv = np.linalg.inv(DFi[:,y] + DFi[:,e].dot(df) + DFi[:,v].dot(Ivy)*self.dZ_Z[0]) 
            return DFhat_Zinv.dot(-DFi[:,Z]),DFhat_Zinv.dot(-DFi[:,Y])
            
        DG = lambda z_i : self.DG(z_i)[nG:,:]
        
        def dY_Z():
            A,B = lambda z_i : dy_Z(z_i)[0], lambda z_i : dy_Z(z_i)[1]
            temp1 = self.integrate(  lambda z_i: DG(z_i)[:,y].dot(A(z_i)) + DG(z_i)[:,Z]  )
            temp2 = self.integrate(  lambda z_i: DG(z_i)[:,y].dot(B(z_i)) + DG(z_i)[:,Y]  )
            return np.linalg.solve(temp2,-temp1)
            
        def residual(dZ_Z):
            self.dZ_Z = dZ_Z
            return (dY_Z()[:nZ]-dZ_Z).flatten()
        
        self.dZ_Z = root(residual,0.9*np.ones(1)).x.reshape(nZ,nZ)
        self.dY_Z = dY_Z()

        def compute_dy_Z(z_i):
            A,B = dy_Z(z_i)
            return A+B.dot(self.dY_Z)
        
        self.dy[Z] = dict_fun(compute_dy_Z)
            
        
            
    def linearize(self):
        '''
        Computes the linearization
        '''
        self.compute_dy()
        DG = lambda z_i : self.DG(z_i)[nG:,:] #account for measurability constraints
        
        def DGY_int(z_i):
            DGi = DG(z_i)
            return DGi[:,Y]+DGi[:,y].dot(self.dy[Y](z_i))
        
        self.DGYinv = np.linalg.inv(self.integrate(DGY_int))
        
        def dYf(z_i):
            DGi = DG(z_i)
            return self.DGYinv.dot(-DGi[:,z]-DGi[:,y].dot(self.dy[z](z_i)))
        
        self.dY = dict_fun(dYf)
        
        self.linearize_aggregate()
        
        self.linearize_parameter()
    
    def linearize_aggregate(self):
        '''
        Linearize with respect to aggregate shock.
        '''
        dy = {}
        
        def Fhat_y(z_i):
            DFi = self.DF(z_i)[:-n,:]
            return DFi[:,y] + DFi[:,v].dot(Ivy).dot(self.dy[z](z_i)).dot(Izy)
            
        Fhat_inv = dict_fun(lambda z_i: -np.linalg.inv(Fhat_y(z_i))) 
        
        def dy_dYprime(z_i):
            DFi = self.DF(z_i)[:-n,:]
            return Fhat_inv(z_i).dot(DFi[:,v]).dot(Ivy).dot(self.dy[Y](z_i))
            
        dy_dYprime = dict_fun(dy_dYprime)
        
        temp = np.linalg.inv( np.eye(nY) - self.integrate(lambda z_i: self.dY(z_i).dot(Izy).dot(dy_dYprime(z_i))) )
        
        self.temp_matrix_Eps = np.eye(nY) - self.integrate(lambda z_i: self.dY(z_i).dot(Izy).dot(dy_dYprime(z_i)))
        
        dYprime = {}
        dYprime[Eps] = temp.dot(
        self.integrate(lambda z_i : self.dY(z_i).dot(Izy).dot(Fhat_inv(z_i)).dot(self.DF(z_i)[:-n,Eps]))        
        )
        dYprime[Y] = temp.dot(
        self.integrate(lambda z_i : self.dY(z_i).dot(Izy).dot(Fhat_inv(z_i)).dot(self.DF(z_i)[:-n,Y] + self.DF(z_i)[:-n,v].dot(Ivy).dot(self.dy[Z](z_i)).dot(IZY)))        
        )
        
        
        dy[Eps] = dict_fun(
        lambda z_i : Fhat_inv(z_i).dot(self.DF(z_i)[:-n,Eps]) + dy_dYprime(z_i).dot(dYprime[Eps])        
        )
        
        dy[Y] = dict_fun(
        lambda z_i : Fhat_inv(z_i).dot(self.DF(z_i)[:-n,Y] + self.DF(z_i)[:-n,v].dot(Ivy).dot(self.dy[Z](z_i)).dot(IZY)) + dy_dYprime(z_i).dot(dYprime[Y])         
        )
        
        #Now do derivatives w.r.t G
        DGi = dict_fun(lambda z_i : self.DG(z_i)[:-nG,:])
        
        DGhat_Y = self.integrate(
        lambda z_i : DGi(z_i)[:,Y] + DGi(z_i)[:,y].dot(dy[Y](z_i))        
        )
        
        self.dY_Eps = -np.linalg.solve(DGhat_Y,self.integrate(
        lambda z_i : DGi(z_i)[:,Eps] + DGi(z_i)[:,y].dot(dy[Eps](z_i))          
        ) )
        
        self.dy[Eps] = dict_fun(
        lambda z_i : dy[Eps](z_i) + dy[Y](z_i).dot(self.dY_Eps)
        )
        
    def linearize_parameter(self):
        '''
        Linearize with respect to aggregate shock.
        '''
        dy = {}
        
        def Fhat_p(z_i):
            DFi = self.DF(z_i)[n:,:]
            df = self.df(z_i)
            return DFi[:,y]+ DFi[:,e].dot(df) + DFi[:,v].dot(Ivy) + DFi[:,v].dot(Ivy).dot(self.dy[z](z_i)).dot(Izy)
            
        Fhat_inv = dict_fun(lambda z_i: np.linalg.inv(Fhat_p(z_i))) 
        
        def dy_dYprime(z_i):
            DFi = self.DF(z_i)[n:,:]
            return Fhat_inv(z_i).dot(DFi[:,v]).dot(Ivy).dot(self.dy[Y](z_i))
            
        dy_dYprime = dict_fun(dy_dYprime)
        
        temp = -np.linalg.inv( np.eye(nY) + self.integrate(lambda z_i: self.dY(z_i).dot(Izy).dot(dy_dYprime(z_i))) )
        
        dYprime = {}
        dYprime[p] = temp.dot(
        self.integrate(lambda z_i : self.dY(z_i).dot(Izy).dot(Fhat_inv(z_i)).dot(self.DF(z_i)[n:,p]))        
        )
        dYprime[Y] = temp.dot(
        self.integrate(lambda z_i : self.dY(z_i).dot(Izy).dot(Fhat_inv(z_i)).dot(self.DF(z_i)[n:,Y]))        
        )
        
        
        dy[p] = dict_fun(
        lambda z_i : -Fhat_inv(z_i).dot(self.DF(z_i)[n:,p]) - dy_dYprime(z_i).dot(dYprime[p])        
        )
        
        dy[Y] = dict_fun(
        lambda z_i : -Fhat_inv(z_i).dot(self.DF(z_i)[n:,Y]) - dy_dYprime(z_i).dot(dYprime[Y])         
        )
        
        #Now do derivatives w.r.t G
        DGi = dict_fun(lambda z_i : self.DG(z_i)[nG:,:])
        
        DGhatinv = np.linalg.inv(self.integrate(
        lambda z_i : DGi(z_i)[:,Y] + DGi(z_i)[:,y].dot(dy[Y](z_i))        
        ))
        
        self.dY_p = -DGhatinv.dot( self.integrate(
        lambda z_i : DGi(z_i)[:,p] + DGi(z_i)[:,y].dot(dy[p](z_i))          
        ) )
        
        self.dy[p] = dict_fun(
        lambda z_i : dy[p](z_i) + dy[Y](z_i).dot(self.dY_p)
        )
        
                              
    def get_df(self,z_i):
        '''
        Gets linear constributions
        '''
        dy = self.dy
        d = {}
        df = self.df(z_i)
        
        d[y,S],d[y,eps],d[y,Z],d[y,Eps] = np.hstack((dy[z](z_i),dy[Y](z_i))),dy[eps](z_i), dy[Z](z_i),dy[Eps](z_i) # first order effect of S and eps on y
    
        d[e,S],d[e,eps],d[e,Z],d[e,Eps] = df.dot(d[y,S]), np.zeros((ne,1)),df.dot(d[y,Z]), np.zeros((ne,1)) # first order effect of S on e
    
        d[Y,S],d[Y,eps],d[Y,Z],d[Y,Eps] = np.hstack(( np.zeros((nY,nz)), np.eye(nY) )),np.zeros((nY,neps)),self.dY_Z,self.dY_Eps
        
        d[z,S],d[z,eps],d[z,Z],d[z,Eps] = np.hstack(( np.eye(nz), np.zeros((nz,nY)) )),np.zeros((nz,neps)),np.zeros((nz,nZ)),np.zeros((nz,1))
        
        d[v,S],d[v,eps],d[v,Z] = Ivy.dot(d[y,S]) + Ivy.dot(dy[Z](z_i)).dot(IZY).dot(d[Y,S]), Ivy.dot(dy[z](z_i)).dot(Izy).dot(dy[eps](z_i)), Ivy.dot(dy[Z](z_i)).dot(self.dZ_Z)
        d[v,Eps] = Ivy.dot(dy[z](z_i).dot(Izy).dot(dy[Eps](z_i)) + dy[Y](z_i).dot(self.dYGamma_Eps) + dy[Z](z_i).dot(IZY).dot(self.dY_Eps) )
        
        d[eps,S],d[eps,eps],d[eps,Z],d[eps,Eps] = np.zeros((neps,nz+nY)),np.eye(neps),np.zeros((neps,nZ)),np.zeros((neps,1))
        
        d[Eps,S],d[Eps,eps],d[Eps,Z],d[Eps,Eps] = np.zeros((1,nz+nY)),np.zeros((1,neps)),np.zeros((1,nZ)),np.eye(1)
        
        d[Z,Z],d[Z,S],d[Z,Eps] = np.eye(nZ),np.zeros((nZ,nY+nz)),np.zeros((nZ,1))

        d[y,z] = d[y,S][:,:nz]
        

        return d
     
    def compute_HFhat(self):
        '''
        Constructs the HFhat functions
        '''
        self.HFhat = {}
        shock_hashes = [eps.__hash__(),Eps.__hash__()]
        for x1 in [S,eps,Z,Eps]:
            for x2 in [S,eps,Z,Eps]:
                
                #Construct a function for each pair x1,x2
                def HFhat_temp(z_i,x1=x1,x2=x2):
                    HF = self.HF(z_i)
                    d = self.get_d(z_i)
                    HFhat = 0.
                    for y1 in [y,e,Y,z,v,eps,Eps]:
                        HFy1 = HF[:,y1,:]
                        for y2 in [y,e,Y,z,v,eps,Eps]:
                            if x1.__hash__() in shock_hashes or x2.__hash__() in shock_hashes:
                                HFhat += quadratic_dot(HFy1[:-n,:,y2],d[y1,x1],d[y2,x2])
                            else:
                                HFhat += quadratic_dot(HFy1[n:,:,y2],d[y1,x1],d[y2,x2])
                    return HFhat
                    
                self.HFhat[x1,x2] = dict_fun(HFhat_temp)
                
    def compute_d2y_ZZ(self):
        '''
        Copute the second derivative of y with respect to Z
        '''
        DF = self.DF
        
        def dy_YZZ(z_i):
            DFi,df = DF(z_i)[n:],self.df(z_i)
            temp = np.linalg.inv(DFi[:,y] + DFi[:,e].dot(df) + self.dZ_Z[0]**2 * DFi[:,v].dot(Ivy))
            
            return - temp.dot( DFi[:,Y] + DFi[:,v].dot(Ivy).dot(self.dy[Z](z_i)).dot(IZY) )
            
        dy_YZZ = dict_fun(dy_YZZ)
        
        def d2y_ZZ(z_i):
            DFi,df,Hf = DF(z_i)[n:],self.df(z_i),self.Hf(z_i)
            dy_Z = self.dy[Z](z_i)
            temp = np.linalg.inv(DFi[:,y] + DFi[:,e].dot(df) + self.dZ_Z[0]**2 * DFi[:,v].dot(Ivy))
            
            return - np.tensordot(temp, self.HFhat[Z,Z](z_i) +np.tensordot(DFi[:,e],quadratic_dot(Hf,dy_Z,dy_Z),1),axes=1)
        d2y_ZZ = dict_fun(d2y_ZZ)
        
        
        def HGhat(z_i,y1,y2):
            HG = self.HG(z_i)[nG:,:]
            d = self.get_d(z_i)
            HGhat = np.zeros((nY,len(y1),len(y2)))
            for x1 in [y,z,Y,Z]:
                HGx1 = HG[:,x1,:]
                for x2 in [y,z,Y,Z]:
                    HGhat += quadratic_dot(HGx1[:,:,x2],d[x1,y1],d[x2,y2])
            return HGhat
                    
        HGhat_ZZ = dict_fun(lambda z_i : HGhat(z_i,Z,Z))
        
        
        temp1 = np.linalg.inv(self.integrate(lambda z_i : self.DG(z_i)[nG:,y].dot(dy_YZZ(z_i)) + self.DG(z_i)[nG:,Y]))
        temp2 = self.integrate(lambda z_i : np.tensordot(self.DG(z_i)[nG:,y],d2y_ZZ(z_i),1) + HGhat_ZZ(z_i))

        
        self.d2Y[Z,Z] = -np.tensordot(temp1,temp2,axes=1)
        
        self.d2y[Z,Z] = dict_fun(lambda z_i: d2y_ZZ(z_i) + np.tensordot(dy_YZZ(z_i),self.d2Y[Z,Z],axes=1) )
        
        #d2y_ZS
        
        def d2y_SZ(z_i):
            DFi,df,Hf = DF(z_i)[n:],self.df(z_i),self.Hf(z_i)
            d,d2y_ZZ = self.get_d(z_i),self.d2y[Z,Z](z_i)
            temp = np.linalg.inv(DFi[:,y] + DFi[:,e].dot(df) + self.dZ_Z[0]*DFi[:,v].dot(Ivy))
            return -np.tensordot(temp, self.HFhat[S,Z](z_i) 
                                       + np.tensordot(DFi[:,e],quadratic_dot(Hf,d[y,S],d[y,Z]),1)
                                       + np.tensordot(DFi[:,v].dot(Ivy), quadratic_dot(d2y_ZZ,IZY.dot(d[Y,S]),self.dZ_Z),1),1) 
        d2y_SZ = dict_fun(d2y_SZ)
        
        def dy_YSZ(z_i):
            DFi,df = DF(z_i)[n:],self.df(z_i)
            temp = np.linalg.inv(DFi[:,y] + DFi[:,e].dot(df) + self.dZ_Z[0]*DFi[:,v].dot(Ivy))
            return - temp.dot(DFi[:,Y] + DFi[:,v].dot(Ivy).dot(self.dy[Z](z_i)).dot(IZY) )
            
        dy_YSZ = dict_fun(dy_YSZ)
        
        HGhat_SZ = dict_fun(lambda z_i : HGhat(z_i,S,Z))
        HGhat_YZ = self.integrate(lambda z_i : (HGhat_SZ(z_i) + np.tensordot(self.DG(z_i)[nG:,y],d2y_SZ(z_i),1))[:,nz:,:])
        HGhat_zZ = lambda z_i : (HGhat_SZ(z_i) + np.tensordot(self.DG(z_i)[nG:,y],d2y_SZ(z_i),1))[:,:nz,:]
        
        temp1 = np.linalg.inv(self.integrate(lambda z_i : self.DG(z_i)[nG:,y].dot(dy_YSZ(z_i)) + self.DG(z_i)[nG:,Y]))
        temp2 = self.integrate(lambda z_i : np.tensordot(self.DG(z_i)[nG:,y],d2y_SZ(z_i),1) + HGhat_SZ(z_i))
        
        def d2Y_zZ(z_i):
            temp1 = np.linalg.inv(self.DG(z_i)[nG:,y].dot(dy_YSZ(z_i)) + self.DG(z_i)[nG:,Y])
            temp2 = HGhat_zZ(z_i) + quadratic_dot(HGhat_YZ,self.dY(z_i),np.eye(nZ))
            return - np.tensordot(temp1,temp2,1)
            
        self.d2Y[z,Z] = dict_fun(d2Y_zZ)
        self.d2Y[Z,z] = lambda z_i : self.d2Y[z,Z](z_i).transpose(0,2,1)
        self.d2y[S,Z] = d2y_SZ
        self.dy[Y,S,Z] = dy_YSZ
        self.d2y[Z,S] = lambda z_i : d2y_SZ(z_i).transpose(0,2,1)
        
        def d2y_Zeps(z_i):
            DFi = DF(z_i)[:-n]
            temp = np.linalg.inv(DFi[:,y] + DFi[:,v].dot(Ivy).dot(self.dy[z](z_i)).dot(Izy))
            return -np.tensordot(temp, self.HFhat[Z,eps](z_i)
                                       + np.tensordot(DFi[:,v].dot(Ivy), 
                                       quadratic_dot(self.d2y[Z,S](z_i)[:,:,:nz],self.dZ_Z,Izy.dot(self.dy[eps](z_i))) ,1),1)
        self.d2y[Z,eps] = dict_fun(d2y_Zeps)
        self.d2y[eps,Z] = lambda z_i : self.d2y[Z,eps].transpose(0,2,1)
        
    def compute_d2y_Eps(self):
        '''
        Computes the 2nd derivatives of y with respect to the aggregate shock.
        '''

        def HGhat(z_i,y1,y2):
            HG = self.HG(z_i)[:-nG,:]
            d = self.get_d(z_i)
            HGhat = np.zeros((nY,len(y1),len(y2)))
            for x1 in [y,z,Y,Z,Eps]:
                HGx1 = HG[:,x1,:]
                for x2 in [y,z,Y,Z,Eps]:
                    HGhat += quadratic_dot(HGx1[:,:,x2],d[x1,y1],d[x2,y2])
            return HGhat        
        
        DF = self.DF
        def dSprime_Eps(z_i):
            return np.vstack((Izy.dot(self.dy[Eps](z_i)),self.dYGamma_Eps))
        
        d2Yprime_EpsEps = self.integrate(lambda z_i : quadratic_dot(self.d2Y[z,z](z_i),Izy.dot(self.dy[Eps](z_i)),Izy.dot(self.dy[Eps](z_i)))
                                            +2*quadratic_dot(self.d2Y[z,Y](z_i),Izy.dot(self.dy[Eps](z_i)),self.dYGamma_Eps)) 
        d2YprimeGZ_EpsEps1 = self.integrate(lambda z_i: quadratic_dot(self.d2Y[z,Z](z_i),Izy.dot(self.dy[Eps](z_i)),IZY.dot(self.dY_Eps)))
        d2YprimeGZ_EpsEps2 = self.integrate(lambda z_i: quadratic_dot(self.d2Y[z,Z](z_i),Izy.dot(self.dy[Eps](z_i)),IZY.dot(self.dYGamma_Eps)))
        
        def DFhat_EpsEps(z_i):
            dSprime_Eps_i,dZprime_Eps = dSprime_Eps(z_i),IZY.dot(self.dY_Eps)
            DFi = DF(z_i)[:-n]
            return self.HFhat[Eps,Eps](z_i) + np.tensordot(DFi[:,v].dot(Ivy),
            quadratic_dot(self.d2y[S,S](z_i),dSprime_Eps_i,dSprime_Eps_i)
            +2*quadratic_dot(self.d2y[S,Z](z_i),dSprime_Eps_i,dZprime_Eps)
            +quadratic_dot(self.d2y[Z,Z](z_i),dZprime_Eps,dZprime_Eps)    
            +np.tensordot(self.dy[Y](z_i),d2Yprime_EpsEps,1)
            +2*np.tensordot(self.dy[Y,S,Z](z_i),d2YprimeGZ_EpsEps1,1) #from dy_GammaZ
            +2*np.tensordot(self.d2y[Y,S,Z](z_i),d2YprimeGZ_EpsEps2,1) #from dy_GammaGamma
            ,axes=1)
        
        def temp(z_i):
            DFi = DF(z_i)[:-n]
            return -np.linalg.inv(DFi[:,y]+DFi[:,v].dot(Ivy).dot(self.dy[z](z_i)).dot(Izy))           
        temp = dict_fun(temp)
        A_i = dict_fun(lambda z_i: np.tensordot(temp(z_i),DFhat_EpsEps(z_i),1))
        B_i = dict_fun(lambda z_i: temp(z_i).dot(DF(z_i)[:-n,Y] + DF(z_i)[:-n,v].dot(Ivy).dot(self.dy[Z](z_i)).dot(IZY) ))
        C_i = dict_fun(lambda z_i: temp(z_i).dot(DF(z_i)[:-n,v].dot(Ivy).dot(self.dy[Y](z_i))))
        A = self.integrate(lambda z_i : np.tensordot(self.dY(z_i).dot(Izy),A_i(z_i),1))
        B,C = self.integrate(lambda z_i: self.dY(z_i).dot(Izy).dot(B_i(z_i))),self.integrate(lambda z_i: self.dY(z_i).dot(Izy).dot(C_i(z_i)))
        
        tempC = np.linalg.inv(np.eye(nY)-C)
        
        d2y_EE =lambda z_i: A_i(z_i) + np.tensordot(C_i(z_i).dot(tempC),A,1)
        dy_YEE = lambda z_i: B_i(z_i) + C_i(z_i).dot(tempC).dot(B)
        
        HGhat_EE = self.integrate(lambda z_i: HGhat(z_i,Eps,Eps)
        + np.tensordot(self.DG(z_i)[:-nG,y],d2y_EE(z_i),1))
        
        DGhat_YEEinv = np.linalg.inv(self.integrate(lambda z_i : self.DG(z_i)[:-nG,Y] + self.DG(z_i)[:-nG,y].dot(dy_YEE(z_i))))
        
        self.d2Y[Eps,Eps] = -np.tensordot(DGhat_YEEinv,HGhat_EE,1) 
        self.d2y[Eps,Eps] = dict_fun(lambda z_i: d2y_EE(z_i) + np.tensordot(dy_YEE(z_i),self.d2Y[Eps,Eps],1))
        
        #Now derivative with respect to sigma_E
        
        def temp2(z_i):
            DFi,df = DF(z_i)[n:],self.df(z_i)
            return -np.linalg.inv(DFi[:,y]+DFi[:,e].dot(df)+DFi[:,v].dot(Ivy)+DFi[:,v].dot(Ivy).dot(self.dy[z](z_i)).dot(Izy))  
        temp2 = dict_fun(temp2)
        
        def A2_i(z_i): #the pure term
            DFi = DF(z_i)[n:]
            df,hf = self.df(z_i),self.Hf(z_i)
            return np.tensordot(temp2(z_i),np.tensordot(DFi[:,e].dot(df),self.d2y[Eps,Eps](z_i),1)
            +np.tensordot(DFi[:,e],quadratic_dot(hf,self.dy[Eps](z_i),self.dy[Eps](z_i)),1)
            ,axes=1)
        A2_i = dict_fun(A2_i)
        B2_i = dict_fun(lambda z_i : temp2(z_i).dot(DF(z_i)[n:,Y] + DF(z_i)[n:,v].dot(Ivy).dot(self.dy[Z](z_i)).dot(IZY) ))
        C2_i = dict_fun(lambda z_i : temp2(z_i).dot(DF(z_i)[n:,v].dot(Ivy).dot(self.dy[Y](z_i))) )
        
        
        A2 = self.integrate(lambda z_i : np.tensordot(self.dY(z_i).dot(Izy),A2_i(z_i),1))
        B2,C2 = self.integrate(lambda z_i: self.dY(z_i).dot(Izy).dot(B2_i(z_i))),self.integrate(lambda z_i: self.dY(z_i).dot(Izy).dot(C2_i(z_i)))
        
        tempC2 = np.linalg.inv(np.eye(nY)-C2)
        
        d2y_sigmaE =lambda z_i: A2_i(z_i) + np.tensordot(C2_i(z_i).dot(tempC2),A2,1)
        dy_YsigmaE = lambda z_i: B2_i(z_i) + C2_i(z_i).dot(tempC2).dot(B2)        
        
        HGhat_sigmaE = self.integrate(lambda z_i: np.tensordot(self.DG(z_i)[nG:,y],d2y_sigmaE(z_i),1))
        
        DGhat_YsigmaEinv = np.linalg.inv(self.integrate(lambda z_i : self.DG(z_i)[nG:,Y] + self.DG(z_i)[nG:,y].dot(dy_YsigmaE(z_i))))
        
        self.d2Y[sigma_E] = - np.tensordot(DGhat_YsigmaEinv,HGhat_sigmaE,1)
        self.d2y[sigma_E] = dict_fun(lambda z_i : d2y_sigmaE(z_i) + np.tensordot(dy_YsigmaE(z_i),self.d2Y[sigma_E],1))
        
    def compute_d2y(self):
        '''
        Computes second derivative of y
        '''
        #DF,HF = self.DF(z_i),self.HF(z_i)
        #df,Hf = self.df(z_i),self.Hf(z_i)
        
        #first compute DFhat, need loadings of S and epsilon on variables
                
        #Now compute d2y
        
        def d2y_SS(z_i):
            DF,df,Hf = self.DF(z_i),self.df(z_i),self.Hf(z_i)
            d = self.get_d(z_i)
            DFi = DF[n:]
            return np.tensordot(np.linalg.inv(DFi[:,y] + DFi[:,e].dot(df) + DFi[:,v].dot(Ivy)),
                -self.HFhat[S,S](z_i) - np.tensordot(DFi[:,e],quadratic_dot(Hf,d[y,S],d[y,S]),1)
                -np.tensordot(DFi[:,v].dot(Ivy),quadratic_dot(self.d2y[Z,Z](z_i),IZY.dot(d[Y,S]),IZY.dot(d[Y,S])),1)
                , axes=1)
        self.d2y[S,S] = dict_fun(d2y_SS)
        
        def d2y_YSZ(z_i):
            DFi,df = self.DF(z_i)[n:],self.df(z_i)
            temp = np.linalg.inv(DFi[:,y] + DFi[:,e].dot(df) + DFi[:,v].dot(Ivy))
            return - np.tensordot(temp, self.dy[Y,S,Z](z_i),1)
        self.d2y[Y,S,Z] = dict_fun(d2y_YSZ)
        
        def d2y_Seps(z_i):
            DF = self.DF(z_i)
            d = self.get_d(z_i)
            DFi = DF[:-n]
            dz_eps= Izy.dot(d[y,eps])
            return np.tensordot(np.linalg.inv(DFi[:,y] + DFi[:,v].dot(Ivy).dot(d[y,z]).dot(Izy)),
            -self.HFhat[S,eps](z_i) - np.tensordot(DFi[:,v].dot(Ivy), self.d2y[S,S](z_i)[:,:,:nz].dot(dz_eps),1)
            -np.tensordot(DFi[:,v].dot(Ivy), quadratic_dot(self.d2y[Z,S](z_i)[:,:,:nz],IZY.dot(d[Y,S]),dz_eps),1)
            , axes=1)
            
        self.d2y[S,eps] = dict_fun(d2y_Seps)
        self.d2y[eps,S] = dict_fun(lambda z_i : self.d2y[S,eps](z_i).transpose(0,2,1))
        
        def d2y_epseps(z_i):
            DF = self.DF(z_i)
            d = self.get_d(z_i)
            DFi = DF[:-n]
            dz_eps= Izy.dot(d[y,eps])
            return np.tensordot(np.linalg.inv(DFi[:,y] + DFi[:,v].dot(Ivy).dot(d[y,z]).dot(Izy)),
            -self.HFhat[eps,eps](z_i) - np.tensordot(DFi[:,v].dot(Ivy), 
            quadratic_dot(self.d2y[S,S](z_i)[:,:nz,:nz],dz_eps,dz_eps),1)
            ,axes=1)
            
        self.d2y[eps,eps] = dict_fun(d2y_epseps)
        
    
        
        
    def quadratic(self):
        '''
        Computes the quadratic approximation
        '''
        self.d2Y,self.d2y = {},{}
        self.dYGamma_Eps = self.integrate(lambda z_i: self.dY(z_i).dot(Izy).dot(self.dy[Eps](z_i)))
        self.get_d = dict_fun(self.get_df)
        self.compute_HFhat()
        self.compute_d2y_ZZ()
        self.compute_d2y()        
        #Now d2Y
        DGhat_f = dict_fun(self.compute_DGhat)
        def DGhat_zY(z_i):
            DGi = self.DG(z_i)[nG:]
            return  (DGhat_f(z_i)[:,:nz,nz:] 
            + np.tensordot(DGi[:,y].dot(self.d2y[Y,S,Z](z_i)), quadratic_dot(self.d2Y[z,Z](z_i),np.eye(nz),IZY),1))
        self.DGhat = {}
        self.DGhat[z,z] = lambda z_i : DGhat_f(z_i)[:,:nz,:nz]
        self.DGhat[z,Y] = dict_fun(DGhat_zY)
        self.DGhat[Y,z] = lambda z_i : self.DGhat[z,Y](z_i).transpose(0,2,1)
        self.DGhat[Y,Y] = self.integrate(lambda z_i : DGhat_f(z_i)[:,nz:,nz:])
        self.compute_d2Y()
        self.compute_dsigma()
        self.compute_d2y_Eps()
        

                
    def compute_d2Y(self):
        '''
        Computes components of d2Y
        '''
        DGhat = self.DGhat
        #First z_i,z_i
        self.d2Y[z,z] = dict_fun(lambda z_i: np.tensordot(self.DGYinv, - DGhat[z,z](z_i),1))
            
        self.d2Y[Y,z] = dict_fun(lambda z_i: np.tensordot(self.DGYinv, - DGhat[Y,z](z_i) 
                            -DGhat[Y,Y].dot(self.dY(z_i))/2. ,1) )
            
        self.d2Y[z,Y] = lambda z_i : self.d2Y[Y,z](z_i).transpose(0,2,1)
        
            
    def compute_DGhat(self,z_i):
        '''
        Computes the second order approximation for agent of type z_i
        '''
        DG,HG = self.DG(z_i)[nG:,:],self.HG(z_i)[nG:,:,:]
        d = self.get_d(z_i)
        d2y = self.d2y
        
        DGhat = np.zeros((nY,nz+nY,nz+nY))
        DGhat += np.tensordot(DG[:,y],d2y[S,S](z_i),1)
        for x1 in [y,Y,z]:
            HGx1 = HG[:,x1,:]
            for x2 in [y,Y,z]:
                DGhat += quadratic_dot(HGx1[:,:,x2],d[x1,S],d[x2,S])
        return DGhat
        
    def compute_d2y_sigma(self,z_i):
        '''
        Computes linear contribution of sigma, dYsigma and dY1sigma
        '''
        DF = self.DF(z_i)
        df,Hf = self.df(z_i),self.Hf(z_i)
        #first compute DFhat, need loadings of S and epsilon on variables
        d = self.get_d(z_i)
        d[y,Y] = d[y,S][:,nz:]
        DFi = DF[n:] #conditions like x_i = Ex_i don't effect this
        
        
        temp = np.linalg.inv(DFi[:,y]+DFi[:,e].dot(df)+DFi[:,v].dot(Ivy+Ivy.dot(d[y,z]).dot(Izy)))
        
        Ahat = (-DFi[:,e].dot(df).dot(self.d2y[eps,eps](z_i).diagonal(0,1,2)) 
        -DFi[:,e].dot(quadratic_dot(Hf,d[y,eps],d[y,eps]).diagonal(0,1,2)) 
        -DFi[:,v].dot(Ivy).dot(d[y,Y]).dot(self.integral_term) )
        
        Bhat = -DFi[:,Y]- DFi[:,v].dot(Ivy).dot(d[y,Z]).dot(IZY)
        Chat = -DFi[:,v].dot(Ivy).dot(d[y,Y])
        return temp.dot(np.hstack((Ahat.reshape(-1,neps),Bhat,Chat)))
        
    def compute_DGhat_sigma(self,z_i):
        '''
        Computes the second order approximation for agent of type z_i
        '''
        DG,HG = self.DG(z_i)[nG:,:],self.HG(z_i)[nG:,:,:]
        d = self.get_d(z_i)
        d2y = self.d2y
        
        DGhat = np.zeros((nY,neps))
        DGhat += DG[:,y].dot(d2y[eps,eps](z_i).diagonal(0,1,2))
        d[eps,eps] =np.eye(neps)
        for x1 in [y,eps]:
            HGx1 = HG[:,x1,:]
            for x2 in [y,eps]:
                DGhat += quadratic_dot(HGx1[:,:,x2],d[x1,eps],d[x2,eps]).diagonal(0,1,2)
        return DGhat
        
    def compute_dsigma(self):
        '''
        Computes how dY and dy_i depend on sigma
        '''
        DG = lambda z_i : self.DG(z_i)[nG:,:]
        #Now how do things depend with sigma
        self.integral_term =self.integrate(lambda z_i:
            quadratic_dot(self.d2Y[z,z](z_i),Izy.dot(self.dy[eps](z_i)),Izy.dot(self.dy[eps](z_i))).diagonal(0,1,2)
            + self.dY(z_i).dot(Izy).dot(self.d2y[eps,eps](z_i).diagonal(0,1,2)))
            
        ABCi = dict_fun(self.compute_d2y_sigma )
        Ai,Bi,Ci = lambda z_i : ABCi(z_i)[:,:neps], lambda z_i : ABCi(z_i)[:,neps:nY+neps], lambda z_i : ABCi(z_i)[:,nY+neps:]
        Atild = self.integrate(lambda z_i: self.dY(z_i).dot(Izy).dot(Ai(z_i)))
        Btild = self.integrate(lambda z_i: self.dY(z_i).dot(Izy).dot(Bi(z_i)))
        Ctild = self.integrate(lambda z_i: self.dY(z_i).dot(Izy).dot(Ci(z_i)))
        tempC = np.linalg.inv(np.eye(nY)-Ctild)

        DGhat = self.integrate(self.compute_DGhat_sigma)
        
        temp1 = self.integrate(lambda z_i:DG(z_i)[:,Y] + DG(z_i)[:,y].dot(Bi(z_i)+Ci(z_i).dot(tempC).dot(Btild)) )
        temp2 = self.integrate(lambda z_i:DG(z_i)[:,y].dot(Ai(z_i)+Ci(z_i).dot(tempC).dot(Atild)) )
        
        self.d2Y[sigma] = np.linalg.solve(temp1,-DGhat-temp2)
        self.d2y[sigma] = dict_fun(lambda z_i: Ai(z_i) + Ci(z_i).dot(tempC).dot(Atild) +
                      ( Bi(z_i)+Ci(z_i).dot(tempC).dot(Btild) ).dot(self.d2Y[sigma]))
                      
    def join_function(self):
        '''
        Joins the data for the dict_maps across functions
        '''
        fast = len(self.dist)%size == 0
        for f in self.dy.values():
            if hasattr(f, 'join'):
                parallel_map_noret(lambda x: f(x[0]),self.dist)
                f.join(fast)
        for f in self.d2y.values():
            if hasattr(f,'join'):       
                parallel_map_noret(lambda x: f(x[0]),self.dist)
                f.join(fast)
          
        self.dY.join(fast)
        
        for f in self.d2Y.values():
            if hasattr(f,'join'):   
                parallel_map_noret(lambda x: f(x[0]),self.dist)
                f.join(fast)
        
        
    def iterate(self,Zbar,quadratic = True):
        '''
        Iterates the distribution by randomly sampling
        '''
        Zhat = Zbar-self.ss.get_Y()[:nZ]
        if rank == 0:
            r = np.random.randn()
            if not shock == None:
                r = shock
            r = min(3.,max(-3.,r))
            E = r*sigma_E
        else:
            E = None
            
        E = comm.bcast(E)
        phat = Para.phat
        Gamma_dist = zip(self.Gamma,self.Gamma_ss)
        
        Y1hat = parallel_sum(lambda z : self.dY(z[1]).dot((z[0]-z[1]))
                          ,Gamma_dist)/len(self.Gamma)
        def Y2_int(x):
            z_i,zbar = x
            zhat = z_i-zbar
            return (quadratic_dot(self.d2Y[z,z](zbar),zhat,zhat) + 2* quadratic_dot(self.d2Y[z,Y](zbar),zhat,Y1hat)).flatten()
        
        Y2hat = parallel_sum(Y2_int,Gamma_dist)/len(self.Gamma)
        
        def Y2_GZ_int(x):
            z_i,zbar = x
            zhat = z_i-zbar
            return quadratic_dot(self.d2Y[z,Z](zbar),zhat,np.eye(nZ)).reshape(nY,nZ)
        Y2hat_GZ =  parallel_sum(Y2_GZ_int,Gamma_dist)/len(self.Gamma)
        
        def compute_ye(x):
            z_i,zbar = x
            extreme = Para.check_extreme(z_i)
            zhat = z_i-zbar
            r = np.random.randn(neps)
            for i in range(neps):
                r[i] = min(3.,max(-3.,r[i]))
            e = r*sigma
            Shat = np.hstack([zhat,Y1hat])
            if not extreme:
                return np.hstack(( self.ss.get_y(zbar).flatten() + self.dy[eps](zbar).dot(e).flatten() + (self.dy[Eps](zbar).flatten())*E
                                    + self.dy[p](zbar).dot(phat).flatten()
                                    + self.dy[z](zbar).dot(zhat).flatten()
                                    + self.dy[Y](zbar).dot(Y1hat).flatten()
                                    + self.dy[Z](zbar).dot(Zhat).flatten()
                                    + 0.5*quadratic*(quadratic_dot(self.d2y[eps,eps](zbar),e,e).flatten() + self.d2y[sigma](zbar).dot(sigma**2).flatten()
                                    +quadratic_dot(self.d2y[S,S](zbar),Shat,Shat) + 2*quadratic_dot(self.d2y[S,eps](zbar),Shat,e)
                                    +self.dy[Y](zbar).dot(Y2hat)      
                                    +2*quadratic_dot(self.d2y[Z,S](zbar),Zhat,Shat)
                                    +2*quadratic_dot(self.d2y[Z,eps](zbar),Zhat,e)
                                    +quadratic_dot(self.d2y[Z,Z](zbar),Zhat,Zhat)
                                    +2*self.dy[Y,S,Z](zbar).dot(Y2hat_GZ).dot(Zhat)
                                    +2*self.d2y[Y,S,Z](zbar).dot(Y2hat_GZ).dot(IZY).dot(Y1hat)
                                    +self.d2y[Eps,Eps](zbar).flatten()*E**2
                                    +self.d2y[sigma_E](zbar).flatten()*sigma_E**2
                                    ).flatten()
                                   ,e))
            else:
                return np.hstack(( self.ss.get_y(zbar).flatten()  + self.dy[Eps](zbar).flatten()*E
                                    + self.dy[p](zbar).dot(phat).flatten()
                                    + self.dy[z](zbar).dot(zhat).flatten()
                                    + self.dy[Y](zbar).dot(Y1hat).flatten()
                                    + self.dy[Z](zbar).dot(Zhat).flatten()
                                    + 0.5*quadratic*(quadratic_dot(self.d2y[eps,eps](zbar),sigma,sigma).flatten() + self.d2y[sigma](zbar).dot(sigma**2).flatten()
                                    +quadratic_dot(self.d2y[S,S](zbar),Shat,Shat)
                                    +self.dy[Y](zbar).dot(Y2hat)      
                                    +2*quadratic_dot(self.d2y[Z,S](zbar),Zhat,Shat)
                                    +quadratic_dot(self.d2y[Z,Z](zbar),Zhat,Zhat)
                                    +2*self.dy[Y,S,Z](zbar).dot(Y2hat_GZ).dot(Zhat)
                                    +2*self.d2y[Y,S,Z](zbar).dot(Y2hat_GZ).dot(IZY).dot(Y1hat)
                                    +self.d2y[Eps,Eps](zbar).flatten()*E**2
                                    +self.d2y[sigma_E](zbar).flatten()*sigma_E**2
                                    ).flatten()
                                   ,0.*e))
        if rank == 0:    
            ye = np.vstack(parallel_map(compute_ye,Gamma_dist))
            y,epsilon = ye[:,:-neps],ye[:,-neps]
            Gamma = y.dot(Izy.T)
            Ynew = (self.ss.get_Y() + Y1hat + self.dY_Eps.flatten() * E
                    + self.dY_p.dot(phat).flatten()
                    + self.dY_Z.dot(Zhat)
                    + 0.5*quadratic*(self.d2Y[sigma].dot(sigma**2) + Y2hat + 2*Y2hat_GZ.dot(Zhat)).flatten()
                    +0.5*(self.d2Y[Eps,Eps].flatten()*E**2 + self.d2Y[sigma_E].flatten()*sigma_E**2))
            Znew = Ynew[:nZ]
            return Gamma,Znew,Ynew,epsilon,y
        else:
            parallel_map(compute_ye,Gamma_dist)
            return None
            
def iterate_ConditionalMean(self,Zbar):
        '''
        Iterates the distribution by randomly sampling
        '''
        Zhat = Zbar-self.ss.get_Y()[:nZ]
        if rank == 0:
            r = np.random.randn()
            if not shock == None:
                r = shock
            r = min(3.,max(-3.,r))
            E = r*sigma_E
        else:
            E = None
            
        E = comm.bcast(E)
        phat = Para.phat
        Gamma_dist = zip(self.Gamma,self.Gamma_ss)
        
        Y1hat = parallel_sum(lambda z : self.dY(z[1]).dot((z[0]-z[1]))
                          ,Gamma_dist)/len(self.Gamma)
        def Y2_int(x):
            z_i,zbar = x
            zhat = z_i-zbar
            return (quadratic_dot(self.d2Y[z,z](zbar),zhat,zhat) + 2* quadratic_dot(self.d2Y[z,Y](zbar),zhat,Y1hat)).flatten()
        
        Y2hat = parallel_sum(Y2_int,Gamma_dist)/len(self.Gamma)
        
        def Y2_GZ_int(x):
            z_i,zbar = x
            zhat = z_i-zbar
            return quadratic_dot(self.d2Y[z,Z](zbar),zhat,np.eye(nZ)).reshape(nY,nZ)
        Y2hat_GZ =  parallel_sum(Y2_GZ_int,Gamma_dist)/len(self.Gamma)
        
        def compute_ye(x):
            z_i,zbar = x
            zhat = z_i-zbar
            r = np.random.randn(neps)
            for i in range(neps):
                r[i] = min(3.,max(-3.,r[i]))
            e = r*sigma
            Shat = np.hstack([zhat,Y1hat])
            return np.hstack(( self.ss.get_y(zbar).flatten() + self.dy[eps](zbar).dot(e).flatten() + (self.dy[Eps](zbar).flatten())*0.
                                    + self.dy[p](zbar).dot(phat).flatten()
                                    + self.dy[z](zbar).dot(zhat).flatten()
                                    + self.dy[Y](zbar).dot(Y1hat).flatten()
                                    + self.dy[Z](zbar).dot(Zhat).flatten()
                                    + 0.5*(quadratic_dot(self.d2y[eps,eps](zbar),e,e).flatten() + self.d2y[sigma](zbar).dot(sigma**2).flatten()
                                    +quadratic_dot(self.d2y[S,S](zbar),Shat,Shat) + 2*quadratic_dot(self.d2y[S,eps](zbar),Shat,e)
                                    +self.dy[Y](zbar).dot(Y2hat)      
                                    +2*quadratic_dot(self.d2y[Z,S](zbar),Zhat,Shat)
                                    +2*quadratic_dot(self.d2y[Z,eps](zbar),Zhat,e)
                                    +quadratic_dot(self.d2y[Z,Z](zbar),Zhat,Zhat)
                                    +2*self.dy[Y,S,Z](zbar).dot(Y2hat_GZ).dot(Zhat)
                                    +2*self.d2y[Y,S,Z](zbar).dot(Y2hat_GZ).dot(IZY).dot(Y1hat)
                                    +self.d2y[Eps,Eps](zbar).flatten()*sigma_E**2
                                    +self.d2y[sigma_E](zbar).flatten()*sigma_E**2
                                    ).flatten()
                                   ,e))
        if rank == 0:    
            ye = np.vstack(parallel_map(compute_ye,Gamma_dist))
            y,epsilon = ye[:,:-neps],ye[:,-neps]
            Gamma = y.dot(Izy.T)
            Ynew = (self.ss.get_Y() + Y1hat + self.dY_Eps.flatten() * 0.
                    + self.dY_p.dot(phat).flatten()
                    + self.dY_Z.dot(Zhat)
                    + 0.5*(self.d2Y[sigma].dot(sigma**2) + Y2hat + 2*Y2hat_GZ.dot(Zhat)).flatten()
                    +0.5*(self.d2Y[Eps,Eps].flatten()*sigma_E**2 + self.d2Y[sigma_E].flatten()*sigma_E**2))
            Znew = Ynew[:nZ]
            return Gamma,Znew,Ynew,epsilon,y
        else:
            parallel_map(compute_ye,Gamma_dist)
            return None
