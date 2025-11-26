# Rotational Integrators for SGs

Unfinished project to implement ideas from the conclusions of my [PhD thesis](https://www.tdx.cat/bitstream/handle/10803/692560/2024_Tesis_Shaw_Luke%20Daniel.pdf?sequence=1).

We seek to consider sampling algorithms which use inexact gradients, in the sense of both convergence and practical performance, as in situations where the accept/reject step of typical MCMC algorithms is too expensive to calculate, the exact gradient may also be too expensive \cite{Leimkuhler2023,Chada2023}. Such a situation occurs in molecular dynamics and machine learning where the potential energy function and its gradient take the forms

$$V(q)=\sum_{i=1}^N V_i(q),\quad \nabla V(q)=\sum_{i=1}^N \nabla V_i(q),$$

with $N$ large, and $V_i(q)=U(q;x_i)+\lambda(q)$ for some functions $U,\lambda:R^{d}\to R$, and data $x_i\in R^d$ . Consequently, an evaluation of the gradient or the potential, as required to perform a numerical integration or calculate the Metropolis acceptance probability, is very expensive ($\mathcal{O}(N)$). Thus one uses an unadjusted algorithm to avoid calculation of the potential, and secondly, using ideas first proposed for stochastic gradient descent \cite{Robbins1951}, one may use, in place of the full gradient, a stochastic approximation \cite{Welling2011}

$$\nabla\widetilde{V}\_{n}(q)= \frac{N}{n} \sum_{i=1}^n \nabla V_{\sigma_i}(q),$$

where $\sigma$ is a random sample of $n$ indices from $\{1,2,3\ldots,N\}$. The cost of a gradient evaluation will thus be $n\ll N$. Convergence for HMC in this unadjusted, inexact gradient case is an area of active study \cite{Chak2023}. 

The basic justification for the validity of the approximate gradient is that, considering the individual gradients  $\nabla V_{\sigma_i}(q)$ as samples, their (scaled) sum should converge, via the Central Limit theorem, to the full gradient $\nabla V(q)$. Extending this idea, via the Bernstein-von Mises theorem of \cref{SplitHMC:BvM} (discussed in greater detail in \cref{sec:BvM}), one expects $\nabla V(q)$ itself to converge (for sufficiently large $N$) to (see \cref{eq:BIBvmThmV}) \cite{Ahn2012}

$$\nabla V(q)\to N\mathcal{I}_F(\hat{q})(q-\hat{q}),$$

where $\mathcal{I}_F(\hat{q})=\mathbb{E}_x\left\[\nabla\_q U(q;x)\nabla_q U(q;x)^T\right\](\hat{q})$. Consequently, one expects an (unbiased) stochastic gradient sample to be distributed
$$\nabla \widetilde{V}_n(q)\sim\mathcal{N}\left(N\mathcal{I}_F(\hat{q})(q-\hat{q}),\frac{N}{n}\mathcal{I}_F(\hat{q})\right).$$
Consequently, one expects $\nabla\widetilde{V}_n(q)-N\mathcal{I}_F(\hat{q})(q-\hat{q})$ to be small, since the size of the noise relative to the mean of the normal distribution scales like $(nN)^{-1/2}$. Hence, one may derive a perturbed semilinear system to which one may apply an (appropriately adapted) RKR integrator. This has been done in an incomplete form for Langevin dynamics (i.e. without momentum $p$) in \cite{Ahn2012}. Moreover, the application of the RKR integrators along similar lines to the modification of HMC used for sampling with stochastic gradient, first considered in \cite{Chen2014}, remains unstudied.
