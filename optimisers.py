# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 10:29:40 2024
https://github.com/GitM0j0/KSGFS/blob/master/sgfs/optim.py
@author: lshaw
"""
import torch

class sampleoptimiser(torch.optim.Optimizer):
    def __init__(self,params,Ifmode='diag',momentum=False,Ndata=1,lam=1,batch_size=32,**kwargs):
        defaults=dict(Ifmode=Ifmode,
                      momentum=momentum,
                      Ndata=Ndata,
                      batch_size=batch_size,
                      lambda_=lam,
                      **kwargs)
        super().__init__(params,defaults)
        self.If=None
        self.qhat=None
        self.Ndata=Ndata
        self.batch_size=batch_size
        self.Ifmode=Ifmode
        self.momentum=momentum
        self._numits=0.
        self._params = self.param_groups[0]["params"]
        self._numel_cache = None
        self._lambda = lam
        self.history=[[] for p in self._params]
        
        def historyhook(opt,*args,**kwargs):
            for i,p in enumerate(opt.get_params()):
                if p.grad is not None:
                    opt.history[i].append(p.detach())
                
        self.register_step_post_hook(historyhook)
        
    def reset(self):
        self._numits=0.
        self.history=[[] for p in self._params]
    
    def get_params(self):
        return self._params
    
    def gradnlogprior(self,q):
        return self._lambda*torch.sign(q) ##divide by self.Ndata???
    
    def _numel(self):
        '''
        Taken from https://pytorch.org/docs/stable/_modules/torch/optim/lbfgs.html#LBFGS
        '''
        if self._numel_cache is None:
            self._numel_cache = sum(
                2 * p.numel() if torch.is_complex(p) else p.numel()
                for p in self._params
            )

        return self._numel_cache
        
    def kappa(self):
        '''
        Increment numits at every call
        '''
        self._numits += 1
        return 1./(self._numits)
    
    def _init_group(self,group,params,grads,momentum_buffer_list):
        for p in group["params"]:
            if p.grad is not None:
                params.append(p)
                grads.append(p.grad)
                
                if self.momentum:
                    state = self.state[p]
                    momentum_buffer_list.append(state.get("momentum_buffer"))
    
    def _update(self, newparams):
        if self.Ifmode=='full':
            offset = 0
            for p in self._params:
                numel = p.numel()
                # view as to avoid deprecated pointwise semantics
                p=newparams[0][offset : offset + numel].view_as(p)
                offset += numel
            assert offset == self._numel()
        else: #for block or diag, self.sample already updated params
            pass
        
    @torch.no_grad
    def step(self,closure=None):
        '''
        Follow LBFGS a bit
        '''
        group = self.param_groups[0]
        kap=self.kappa()
        params,grads,momentum_buffer_list=[],[],[]
        self._init_group(group, params, grads, momentum_buffer_list)

        if self.Ifmode=='full':
            grads=[torch.cat([p.view((-1)) for p in grads])]
            params=[torch.cat([p.view((-1))for p in params])]
            momentum_buffer_list=[torch.cat([p.flatten() for p in momentum_buffer_list])]
            if self.If==None:
                self.If=[torch.cov(grads)]
                self.qhat=[params]
        else: #diag or block mode
            cv=[torch.var(g,dim=0) for g in grads] if self.Ifmode=='diag' else [torch.cov(g) for g in grads]
            if self.If==None:
                self.If=[c for c in cv]
                self.qhat=[p for p in params]
                
        for i in range(len(params)):
            cv=torch.var(grads[i],dim=0) if self.Ifmode=='diag' else torch.cov(grads[i])
            self.If[i]=(1.-kap)*self.If[i]+kap*cv
            self.qhat[i]=(1.-kap)*self.qhat[i]+kap*params[i]
        self.sample(grads,params,momentum_buffer_list,closure)
        self._update(params) #Ifmode full means only updated flattened copy of params
        
        if self.momentum:
            # update momentum_buffers in state
            for p, momentum_buffer in zip(params, momentum_buffer_list):
                state = self.state[p]
                state["momentum_buffer"] = momentum_buffer

    
class SGFS(sampleoptimiser):
    def __init__(self,params,Ifmode,Ndata,batch_size,alpha=lambda t:.01):
        self.gamma=torch.tensor((self.batch_size+self.Ndata)/self.batch_size)
        super().__init__(params,Ifmode=Ifmode,momentum=False,Ndata=Ndata,batch_size=batch_size,alpha=alpha,gamma=self.gamma)
        self.alpha = lambda : alpha(self._numits) #can be function of self._numits
        if self.Ifmode=='diag':
            self.conditioner = lambda g,L: g/L**2/(self.Ndata*self.gamma)
            self.noiseconditioner = lambda noise,L: noise/(L*torch.sqrt(self.Ndata*self.gamma))
        else:
            self.conditioner = lambda g,L: torch.cholesky_solve(g,L)/(self.Ndata*self.gamma)
            self.noiseconditioner = lambda noise,L: torch.linalg.solve_triangular(L.T,noise)/(torch.sqrt(self.Ndata*self.gamma))
    
    def sample(self,grad,q,momentum,closure=None):
        ## B always equal to gamma*I_N, If=LL^T
        # p is dummy
        a=self.alpha()
        for i in range(len(grad)):
            grad_=grad[i]+self.gradnlogprior(q[i])
            if self.Ifmode=='diag':
                L=self.If[i].sqrt()
            else: 
                L=torch.cholesky(self.If[i])
            noise=a*self.noiseconditioner(torch.randn_like(q[i]),L)
            conditioned=self.conditioner(grad_,L)
            q[i].add_(2*(noise+conditioned)/(1.+a**2))
    
class SGHMC(sampleoptimiser):
    def __init__(self,params,Ifmode,Ndata,bs,m=50,alpha=0.01,beta=0.,eta=.2e-5):
        super().__init__(params,Ifmode=Ifmode,momentum=True,
                         Ndata=Ndata,batch_size=bs,m=m,alpha=alpha,beta=beta,eta=eta)
        self.m=m #number of steps to integrate between samples
        self.alpha=torch.tensor(alpha)
        self.beta=torch.tensor(beta)
        self.eta=torch.tensor(eta)
        self.momentumdist = lambda v : self.eta*torch.randn_like(v)

    def sample(self,grad,q,momentum,closure):
        v = momentum
        for i in range(len(grad)):
            if v[i] is None:
                v[i] = self.momentumdist(grad[i])
            else:
                v[i] = self.momentumdist(v[i]) # does nothing if don't want to resample momentum
        for _ in range(self.m):
            for i in range(len(grad)):
                grad_=grad[i]+self.gradnlogprior(q[i])
                q[i]+=v[i]
                noise=torch.sqrt(2*self.eta*(self.alpha-self.beta))*torch.randn_like(v[i])
                v[i]+=-self.alpha*v[i]+self.eta*grad_+noise
            with torch.enable_grad():
                loss = float(closure()) #recalculate gradients
        momentum = v
        return q

class SGLDFS(sampleoptimiser):
    def __init__(self,params,Ifmode,Ndata,bs,h=lambda t: 0.01):
        self.gamma=torch.tensor((self.batch_size+self.Ndata)/self.batch_size)
        super().__init__(params,Ifmode=Ifmode,momentum=False,Ndata=Ndata,batch_size=bs,h=h,gamma=self.gamma)
        self.h = lambda : h(self._numits) #can be function of _numits
        if self.Ifmode=='diag':
            self.conditioner = lambda g,L: g/L**2/(self.Ndata)
            self.noiseconditioner = lambda noise,L: noise/(L*torch.sqrt(self.Ndata))
        else:
            self.conditioner = lambda g,L: torch.cholesky_solve(g,L)/(self.Ndata)
            self.noiseconditioner = lambda noise,L: torch.linalg.solve_triangular(L.T,noise)/(torch.sqrt(self.Ndata))
    
    def sample(self,grad,q,momentum,closure=None):
        ## B always equal to gamma*I_N, If=LL^T
        h_=self.h()
        for i in range(len(grad)):
            partial_grad=grad[i]+self.gradnlogprior(q[i])+self.Ndata*self.If[i]@(q[i]-self.qhat[i])
            if self.Ifmode=='diag':
                L=self.If[i].sqrt()            
            else:
                L=torch.cholesky(self.If[i])
            noise=torch.sqrt(1.-torch.exp(-2*h_))*self.noiseconditioner(torch.randn_like(q[i]),L)
            conditioned=(1.-torch.exp(-h_))*self.conditioner(partial_grad,L)
            q[i].add_(-self.qhat[i])
            q[i].mul_(torch.exp(-h_))
            q[i].add_(conditioned+noise)
            q[i].add_(self.qhat[i])

class SGHMCFS(sampleoptimiser):
    def __init__(self,params,Ifmode,Ndata,bs,h=lambda t: 0.01,customsampler=None):
        super().__init__(params,Ifmode=Ifmode,momentum=True,Ndata=Ndata,batch_size=bs,h=h)
        self.h = lambda : h(self._numits) #can be made function of numits
        if self.Ifmode=='diag':
            self.conditioner = lambda g,L: g/L**2/(self.Ndata)
            self.noiseconditioner = lambda noise,L: noise/(L*torch.sqrt(self.Ndata))
        else:
            self.conditioner = lambda g,L: torch.cholesky_solve(g,L)/(self.Ndata)
            self.noiseconditioner = lambda noise,L: torch.linalg.solve_triangular(L.T,noise)/(torch.sqrt(self.Ndata))
        if customsampler is None:
            self.mysampler=self.SEEuler
            self.samplerstr='StochasticExponentialEuler'
        else:
            self._set_sampler(customsampler)
        self.defaults['sampler']=self.samplerstr

    def _set_sampler(self,s):
        sampler=[]
        for l in s:
            match l:
                case 'R':
                    sampler.append(self.R)
                case 'O':
                    sampler.append(self.O)
                case 'U':
                    sampler.append(self.U)
                case 'B':
                    sampler.append(self.B)
                case '_':
                    raise ValueError('Custom sampler can only include R,O,U,B.')
        self.mysampler= lambda q,p,pg,L,h: [f(q,p,pg,L,h) for f in sampler]
        self.samplerstr=s
        
    def _get_sampler(self):
        print(self.samplerstr)
        
    def U(self,q,p,pg,L,h_):
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

        noise1=self.noiseconditioner(torch.randn_like(q),L)
        noise2=self.noiseconditioner(torch.randn_like(p),L)
        #Step 2: add correlated noise
        q.add_(f1*noise1)
        p.add_(f2*noise1+f3*noise2)    
    
    def R(self,q,p,pg,L,h):
        c,s=torch.cos(h),torch.sin(h)
        q,p=c*q+s*p,-s*q+c*p
    
    def O(self,q,p,pg,L,h):
        noise=self.noiseconditioner(torch.randn_like(p),L)
        e=torch.exp(-h)
        p.mul_(e).add_(noise,alpha=torch.sqrt(1-e**2))
    
    def B(self,q,p,pg,L,h):
        conditioned=self.conditioner(pg,L)
        p.add_(conditioned,alpha=h)
    
    def SEEuler(self,q,p,pg,L,h_):
        '''
        Stochastic exponential Euler scheme
        '''
        f=torch.sqrt(torch.tensor(3))/2
        e=torch.exp(-h_/2)
        s=e*torch.sin(f*h_)/f
        sm=e*torch.sin(f*h_-2*torch.pi/3)/f
        
        #e^hA + noise
        self.U(q, p, pg, L, h_)
        
        #Ainv(1-e^hA)*grad
        conditioned=self.conditioner(pg,L)
        q.add_(conditioned,alpha=(1.+sm))
        p.add_(conditioned,alpha=-s)
        
    def sample(self,grad,q,momentum,closure=None):
        ## B always equal to gamma*I_N, If=LL^T
        h_=self.h()
        for i in range(len(grad)):
            partial_grad=grad[i]+self.gradnlogprior(q[i])+self.Ndata*self.If[i]@(q[i]-self.qhat[i])
            if self.Ifmode=='diag':
                L=self.If[i].sqrt()
            else:
                L=torch.cholesky(self.If[i])
            q[i]-=self.qhat[i]
            self.mysampler(q[i], momentum[i], partial_grad,L,h_)
            q[i]+=self.qhat[i]

    