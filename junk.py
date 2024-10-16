# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 20:57:12 2024
https://github.com/GitM0j0/KSGFS/blob/master/sgfs/optim.py
@author: lshaw
"""
import torch

class GenAlg:
    def __init__(self,Experiment):
        self.Experiment=Experiment
        self.dataloader=torch.utils.data.DataLoader(Experiment.dataset, batch_size=Experiment.bs, shuffle=True)
        self.dataiter = iter(self.dataloader)
        self.Ndata=len(self.dataloader.dataset)
        self.If=None
        self.qhat=None
        self.kappa = lambda t: 1/t
        self.diag=True
        
    def stochgrad(self,q,t):
        data, labels = next(self.dataiter)
        nlogprior,nll=self.Experiment.gradnlogprior(q),self.Experiment.gradnll(q,data,labels)
        CV=torch.var(nll,dim=0) if self.diag else torch.cov(nll) #for nll rows=variables, cols=observations
        self.If=(1.-self.kappa(t))*self.If+self.kappa(t)*CV
        self.qhat=(1.-self.kappa(t))*self.qhat+self.kappa(t)*q
        return nlogprior,nll
        
    def update_step(self,q,p,t):
        return NotImplementedError()
    
    def loop(self,q0,I0,T,warmstart=False):
        q=q0.copy()
        p=torch.randn_like(q)
        if not warmstart:
            self.If=I0.copy()
        samples=torch.zeros((T,*q.shape))
        for t in range(T):
            q,p=self.update_step(q,p,t)
            samples[t]=q
        return samples
    
class SFGS(GenAlg):
    def __init__(self,Experiment):
        super().__init__(Experiment)
        self.gamma=(self.dataloader.batch_size+self.Ndata)/self.dataloader.batch_size
        self.alpha = lambda t: .01
        self.momentumdist= lambda q : 0. 
        
    def update_step(self,q,p,t):
        ## B always equal to gamma*I_N, If=LL^T
        # p is dummy
        nlogprior,nll=self.stochgrad(q,t)
        grad=self.Ndata*nll.mean(dim=0)+nlogprior
        a=self.alpha(t)
        L=self.getL()
        if self.diag:
            L=self.If.sqrt()
            noise=a*torch.sqrt(self.gamma*self.Ndata)*L*torch.randn_like(q)
            conditioned=(grad+noise)/L**2
        else: 
            L=torch.cholesky(self.If)
            noise=a*torch.sqrt(self.gamma*self.Ndata)*L@torch.randn_like(q)
            conditioned=torch.cholesky_solve((grad+noise),L)
            
        return q+2*conditioned/(self.gamma*self.Ndata*(1.+a**2)), #empty p return
    
class SGHMC(GenAlg):
    def __init__(self,Experiment):
        super().__init__(Experiment)
        self.m=50 #number of steps to integrate between samples
        self.alpha=torch.tensor(0.01)
        self.beta=torch.tensor(0.)
        self.eta=torch.tensor([0.2*1e-5])
        self.momentumdist = lambda v : self.eta*torch.randn_like(v)

    def update_step(self,q,v,t):
        v=self.momentumdist(v) # does nothing if don't want to resample momentum
        for _ in range(self.m):
            nlogprior,nll=self.stochgrad(q,t)
            grad=self.Ndata*nll.mean(dim=0)+nlogprior
            q+=v
            noise=torch.sqrt(2*self.eta*(self.alpha-self.beta))*torch.randn_like(v)
            v+=-self.alpha*v+self.eta*grad+noise
        return q,v
    
class SGLDFS(GenAlg):
    def __init__(self,Experiment):
        super().__init__(Experiment)
        self.gamma=(self.dataloader.batch_size+self.Ndata)/self.dataloader.batch_size
        self.h = lambda t : .01
    
    def update_step(self,q,p,t):
        ## B always equal to gamma*I_N, If=LL^T
        # p is dummy
        nlogprior,nll=self.stochgrad(q,t)
        partial_grad=self.Ndata*nll.mean(dim=0)+nlogprior+self.Ndata*self.If@(q-self.qhat)
        h_=self.h(t)
        noise=torch.randn_like(q)
        if self.diag:
            L=self.If.sqrt()
            noise*=torch.sqrt((1.-torch.exp(-2*h_))/self.Ndata)/L
            conditioned=(1.-torch.exp(-h_))*partial_grad/L**2
        else:
            L=torch.cholesky(self.If)
            noise=torch.sqrt((1.-torch.exp(-2*h_))/self.Ndata)*torch.linalg.solve_triangular(L.T,noise)
            conditioned=(1.-torch.exp(-h_))*torch.cholesky_solve(partial_grad,L)/self.Ndata
        q-=self.qhat
        q*=torch.exp(-h_)
        q+=conditioned+noise
        q+=self.qhat
        return q,


class SGHMCFS(GenAlg):
    def __init__(self,Experiment):
        super().__init__(Experiment)
        self.gamma=(self.dataloader.batch_size+self.Ndata)/self.dataloader.batch_size
        self.h= lambda t: 0.01

    def U(self,q,p,h_):
        f=torch.sqrt(torch.tensor(3))/2
        e=torch.exp(-h_/2)
        s=e*torch.sin(f*h_)/f
        sp=e*torch.sin(f*h_+2*torch.pi/3)/f
        sm=e*torch.sin(f*h_-2*torch.pi/3)/f
        f1=torch.sqrt(1-4/3*(s**2+sm**2))
        f2=-4/3*s*(sp+sm)/f1
        f3=torch.sqrt(1-4/3*(s**2+sp**2)-f2**2)
        
        ##Step 1: e^hA
        q,p=-q*sm+s*p,-s*q+sp*p

        noise1=torch.randn_like(q)
        noise2=torch.randn_like(p)
        if self.diag:
            L=self.If.sqrt()
            noise1/=(L*torch.sqrt(self.Ndata))
            noise2/=(L*torch.sqrt(self.Ndata))
        else:
            L=torch.cholesky(self.If)
            noise1=torch.linalg.solve_triangular(L.T,noise1)/(torch.sqrt(self.Ndata))
            noise2=torch.linalg.solve_triangular(L.T,noise2)/(torch.sqrt(self.Ndata))
        
        #Step 2: add correlated noise
        return q+f1*noise1,p+f2*noise1+f3*noise2       
    
    def R(self,q,p,h):
        c,s=torch.cos(h),torch.sin(h)
        return c*q+s*p,-s*q+c*p
    
    def O(self,q,p,h):
        noise=torch.randn_like(p)
        if self.diag:
            L=self.If.sqrt()
            noise/=(L*torch.sqrt(self.Ndata))
        else:
            L=torch.cholesky(self.If)
            noise=torch.linalg.solve_triangular(L.T,noise)/(torch.sqrt(self.Ndata))
        
        e=torch.exp(-h)
        return q,e*p+torch.sqrt(1-e**2)*noise
    
    def B(self,q,p,t,conditioned):
        h=self.h(t)
        return q,p+h*conditioned
    
    def SEEuler(self,q,p,t,conditioned):
        '''
        Stochastic exponential Euler scheme
        '''
        h_=self.h(t)
        f=torch.sqrt(torch.tensor(3))/2
        e=torch.exp(-h_/2)
        s=e*torch.sin(f*h_)/f
        sm=e*torch.sin(f*h_-2*torch.pi/3)/f
        
        q-=self.qhat
        
        #e^hA + noise
        q,p=self.U(q, p, h_)
        
        #Ainv(1-e^hA)*grad
        q+=conditioned*(1.+sm)
        p-=s*conditioned
        
        q+=self.qhat
        return q,p
        
    def update_step(self,q,p,t):
        ## B always equal to gamma*I_N, If=LL^T
        # p is dummy
        nlogprior,nll=self.stochgrad(q,t)
        partial_grad=self.Ndata*nll.mean(dim=0)+nlogprior+self.Ndata*self.If@(q-self.qhat)
        if self.diag:
            L=self.If.sqrt()
            conditioned=partial_grad/L**2/(self.Ndata)
        else:
            L=torch.cholesky(self.If)
            conditioned=torch.cholesky_solve(partial_grad,L)/(self.Ndata)

        return self.stochasticEuler(q, p, t, conditioned)
    
    