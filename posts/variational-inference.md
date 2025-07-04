---
title: "Revisiting Variational Inference for Statististican"
subtitle: "In this blogpost, I'll extend the derivations of the Gaussian Mixture model of the paper in the hope to elucidate some of the steps over which the authors went quickly."
date: "2020-11-19"
published: true
---

Blei et. al. illustrate the coordinate ascent variational inference (CAVI) using a simple Gaussian Mixture model. The model[^1] places a prior on the mean of each component while keeping the variance of the likelihood fixed.

$$
\begin{aligned}
\mu_{k} & \sim \mathcal{N}\left(0, \sigma^{2}\right) \\
\mathbf{z}_{n} & \sim \text { Categorical }(1 / K, \dots, 1 / K) \\
x_{n} \mid \mathbf{z}_{n}, \boldsymbol{\mu} & \sim \mathcal{N}\left(\mathbf{z}_{n}^{\top}\boldsymbol{\mu}, 1\right)
\end{aligned}
$$

In the following, we will derive the joint probability and CAVI update equations for the model. Finally, we use these equations to implement the model in Python.

## Constructing the log joint

We start by defining the components of the model. Note that we can write the probability of the prior component means as

$$
p(\boldsymbol{\mu})=\prod_k \mathcal{N}(\mu_k|0, \sigma^2).
$$

Similarly, the prior for the latent variables $\mathbf{z}_n$ may be expressed as

$$
p(\mathbf{z}_{n})=\prod_k \left(\frac{1}{K}\right)^{z_{nk}}
$$

while the likelihood is given by

$$
p(x_n|\boldsymbol{\mu}, \mathbf{z}_{n})=\prod_k \mathcal{N}(0|\mu_k, 1)^{z_{nk}}.
$$

We now introduce the variables $\mathbf{X} = \{x_n\}_{n=1}^{N}$ and $\mathbf{Z}=\{ \mathbf{z}_n\}_{n=1}^{N}$ to denote the complete dataset. Note that $p(\mathbf{Z})$ and $p(\mathbf{X}\|\boldsymbol{\mu}, \mathbf{Z})$ are simply

$$
p(\mathbf{Z})=\prod_n\prod_k \left(\frac{1}{K}\right)^{z_{nk}}\quad\text{and}\quad p(\mathbf{X}|\boldsymbol{\mu}, \mathbf{Z})=\prod_n \prod_k \mathcal{N}(0|\mu_k, 1)^{z_{nk}}.
$$

With these equations we can construct the joint distribution which factorizes as follows

$$
p(\mathbf{X}, \boldsymbol{\mu}, \mathbf{Z})= p(\boldsymbol{\mu}) p(\mathbf{X}|\boldsymbol{\mu}, \mathbf{Z}) p(\mathbf{Z})= \prod_k \mathcal{N}(\mu_k|0, \sigma^2) \prod_n\prod_k \left(\frac{1}{K}\cdot \mathcal{N}(0|\mu_k, 1)\right)^{z_{nk}}.
$$

Finally, we end up with the following log joint distribution for the model

$$
\log{p(\mathbf{X}, \boldsymbol{\mu}, \mathbf{Z})} = \sum_k \log{\mathcal{N}(\mu_k|0, \sigma^2)} +\sum_n\sum_k z_{nk} \left(\log{\frac{1}{K}}+ \log{\mathcal{N}(0|\mu_k, 1)}\right).\tag{1}
$$

## The variational density for the mixture assignments

To obtain the (log) variational distribution of $\mathbf{z}_n$, we simply take the expectation of the log joint $(1)$ with respect to all other variables of the model. In our simple Gaussian mixture model this corresponds to $q(\mu_k)$, as it is the only other variable of the model.

$$
 \begin{aligned}
 \log q^{*}\left(\mathbf{z}_{n}\right) &=\mathbb{E}_{q(\mu_k)}[\log p(x_n, \boldsymbol{\mu}, \mathbf{z}_n)] +\text { const. } \\
 &=\mathbb{E}_{q(\mu_k)}\left[\log p\left(x_{n} | \boldsymbol{\mu}, \mathbf{z}_{n}\right)+\log p\left(\mathbf{z}_{n}\right)\right]+\text { const. } \\
 &=\mathbb{E}_{q(\mu_k)}\left[\sum_{k} z_{nk}\left(\log \frac{1}{K}+\log \mathcal{N}\left(0 \mid \mu_{k}, 1\right)\right)\right]+\operatorname{const.}  \\
 &=\mathbb{E}_{q(\mu_k)}\left[-\cancel{\sum_{k} z_{n k} \log \frac{1}{K}}+\sum_{k} z_{n k}\left(-\frac{1}{2} \log 2 \pi-\frac{1}{2}\left(x_{n}-\mu_{k}\right)^{2}\right)\right] +\operatorname{const.} \\
 &=\mathbb{E}_{q(\mu_k)}\left[-\cancel{\sum_{k}  \frac{z_{n k}}{2} \log 2 \pi} -\sum_{k}  \frac{z_{n k}}{2}\left(x_{n}^2-2x_n\mu_k+\mu_{k}^2\right)\right] +\operatorname{const.} \\
 &=\mathbb{E}_{q(\mu_k)}\left[-\sum_{k}  \cancel{\frac{z_{n k}}{2} x_{n}^2} - z_{n k} x_n\mu_k+ \frac{z_{n k}}{2} \mu_{k}^2\right] +\operatorname{const.} \\
 &=\sum_{k} z_{n k} x_n\mathbb{E}_{q(\mu_k)}[\mu_k] - \frac{z_{n k}}{2} \mathbb{E}_{q(\mu_k)}[\mu_{k}^2] +\operatorname{const.} \\
 &=\sum_{k} z_{n k} \left(x_n\mathbb{E}_{q(\mu_k)}[\mu_k] - \frac{1}{2} \mathbb{E}_{q(\mu_k)}[\mu_{k}^2]\right) +\operatorname{const.} \\
 &=\sum_{k} z_{n k} \log{\rho_{nk}} +\operatorname{const.} \tag{2}
 \end{aligned}
$$

Here I have canceled constant terms in $z_{nk}$ (only terms including the expectations w.r.t. to $q(\mu_k)$ change). Let's take a closer look at the last line of $(2)$; exponentiating reveals $\log q^{*}(\mathbf{z}_n)$ that it has the form of a multinomial distribution

$$
q^{*}\left(\mathbf{z}_{n}\right)\propto \prod_{k} \rho_{nk} ^ {z_{n k}},
$$

thus in order to normalise the distribution, we require that the variational parameter $\rho_{nk}$ represents a probability. We therefore define

$$
r_{nk} = \frac{\rho_{nk}}{\sum_j \rho_{nj}} = \frac{e^{x_n\mathbb{E}_{q(\mu_k)}[\mu_k] - \frac{1}{2} \mathbb{E}_{q(\mu_k)}[\mu_{k}^2]}}{\sum_j e^{x_n\mathbb{E}_{q(\mu_j)}[\mu_j] - \frac{1}{2} \mathbb{E}_{q(\mu_j)}[\mu_{j}^2]}}
$$

and the our final density is given by

$$
q^{*}\left(\mathbf{z}_{n};\mathbf{r}_n\right) = \prod_{k} r_{nk} ^ {z_{n k}}.\tag{3}
$$

## The variational density for the means

We proceed similarly to determine the variational density of $q(\mu_k)$

$$
 \begin{aligned}
 \log q^{*}\left(\mathbf{\mu}_{k}\right) &=\mathbb{E}_{q(\mathbf{z}_n)}[\log p(\mathbf{X}, \boldsymbol{\mu}, \mathbf{Z})] +\text { const. } \\
 &=\mathbb{E}_{q(\mathbf{z}_n)}\left[\log p\left(\boldsymbol{\mu}\right) + \log p\left(\mathbf{X} | \boldsymbol{\mu}, \mathbf{Z}\right)\right]+\text { const. } \\
 &=\mathbb{E}_{q(\mathbf{z}_n)}\left[\log{\mathcal{N}(\mu_k|0, \sigma^2)}+\sum_{n} z_{nk} \log \mathcal{N}\left(0 \mid \mu_{k}, 1\right)\right]+\operatorname{const.}  \\
 &=\mathbb{E}_{q(\mathbf{z}_n)}\left[-\cancel{\frac{1}{2}\log{2\pi\sigma^2}}-\frac{1}{2\sigma^2}\mu_k^2+ \sum_{n} z_{n k}\left(\cancel{-\frac{1}{2} \log 2 \pi}-\frac{1}{2}\left(x_{n}-\mu_{k}\right)^{2}\right)\right] +\operatorname{const.} \\
 &=\mathbb{E}_{q(\mathbf{z}_n)}\left[-\frac{1}{2\sigma^2}\mu_k^2 -\sum_{n}  \frac{z_{n k}}{2}\left(x_{n}^2-2x_n\mu_k+\mu_{k}^2\right)\right] +\operatorname{const.} \\
 &=-\frac{1}{2\sigma^2}\mu_k^2 +\mathbb{E}_{q(\mathbf{z}_n)}\left[-  \cancel{\sum_{n}\frac{z_{n k}}{2} x_{n}^2} + \mu_k\sum_{n} z_{n k} x_n - \mu_{k}^2\sum_{n}\frac{z_{n k}}{2} \right] +\operatorname{const.} \\
 &=-\frac{1}{2\sigma^2}\mu_k^2 + \mu_k\sum_{n} \mathbb{E}_{q(\mathbf{z}_n)}[z_{n k}] x_n - \mu_{k}^2\sum_{n}\frac{\mathbb{E}_{q(\mathbf{z}_n)}[z_{n k}]}{2} +\operatorname{const.} \\
 &= \mu_k\sum_{n} \mathbb{E}_{q(\mathbf{z}_n)}[z_{n k}] x_n - \mu_{k}^2(\sum_{n}\frac{\mathbb{E}_{q(\mathbf{z}_n)}[z_{n k}]}{2}+\frac{1}{2\sigma^2}) +\operatorname{const.} \\
 &=\begin{bmatrix} \mu_k \\ \mu_k^2 \end{bmatrix}^T\begin{bmatrix} \mathbb{E}_{q(\mathbf{z}_n)}[z_{n k}] x_n \\ -(\frac{1}{2}\sum_{n}\mathbb{E}_{q(\mathbf{z}_n)}[z_{n k}]+\frac{1}{\sigma^2}) \end{bmatrix} +\operatorname{const.}
 \end{aligned}
$$

The last line of the derivation suggests that the variational distribution for $\mu_k$ is Gaussian with natural parameter $\boldsymbol{\eta}=[\mathbb{E}_{q(\mathbf{z}_n)}[z_{n k}] x_n, -(\sum_{n}\frac{\mathbb{E}_{q(\mathbf{z}_n)}[z_{n k}]}{2}+\frac{1}{2\sigma^2})]$ and sufficient statistic $t(\mu_k)=[\mu_k, \mu_k^2]$. Using standard formulas {% cite blei2016exponential %}, we find that the mean posterior mean and covariance are given by

$$
s^2_k=-\frac{1}{2\eta_2}=\frac{1}{\sum_{n}\mathbb{E}_{q(\mathbf{z}_n)}[z_{n k}]+\frac{1}{\sigma^2}}\quad\text{and}\quad m_k=\eta_1\cdot s_k^2=\frac{\mathbb{E}_{q(\mathbf{z}_n)}[z_{n k}] x_n}{\sum_{n}\mathbb{E}_{q(\mathbf{z}_n)}[z_{n k}]+\frac{1}{\sigma^2}}.\tag{4}
$$

## Solving expectations

Although we have derived parameters of our variational distributions, we can't work properly with the results as all of them contain unresolved expectations. However, we can leverage the form of our variational distributions, i.e. $z_{nk}$ and $\mu_k$ are respectively multinomial and normally distributed. For example, to solve the expectation of $z_{nk}$, we use $(3)$ to determine

$$
\mathbb{E}_{q_(\mathbf{z}_n)}[z_{nk}]=\sum_{\mathbf{z}}\mathbf{z}_n q^{*}(\mathbf{z}_n; r_n)=\sum_{\mathbf{z}}\mathbf{z}_n \prod_{k} r_{nk} ^ {z_{n k}} = r_{nk}.\tag{5}
$$

Now we can simply plug $(5)$ into $(4)$ to obtain

$$
\sigma^2_N=\frac{1}{\sum_{n}r_{nk}+\frac{1}{\sigma^2}}\quad\text{and}\quad\mu_N=\frac{r_{nk} x_n}{\sum_{n}r_{nk}+\frac{1}{\sigma^2}}.
$$

It is easy to see that $\mathbb{E}_{q(\mu_k)}[\mu_k]=m_k$. To determine the second moment of $\mu_k$, which is also required to compute $r_{nk}$, we make use of standard properties of the variance[^2]

$$
\mathbb{E}_{q(\mu_k)}[\mu_k^2]=m_k^2+s_k^2.
$$

## Implementing the model

With these equation in hand we can easily implement the model.

```python
class GaussianMixtureCavi:
    def __init__(self, X, K):
        self.X = X
        self.K = K
        self.m = np.random.uniform(np.min(X), np.max(X), K)
        self.s = np.random.normal(size=K) \*\* 2
        self.σ = 1

    def fit(self):
        for it in range(100):
            y = self.X.reshape(-1, 1) * self.m.reshape(1, -1) - (
                0.5 * (self.s + self.m**2)
            ).reshape(1, -1)
            α = np.max(y, 1).reshape(-1, 1)
            self.ϕ = np.exp(y - (α + np.log(np.exp(y - α).sum(1, keepdims=True))))
            denom = 1 / self.σ + self.ϕ.sum(0, keepdims=True)
            self.m = (self.ϕ * self.X.reshape(-1, 1)).sum(0) / denom
            self.s = 1 / denom

    def approx_mixture(self, x):
        return np.stack(
            [
                ϕ_i * stats.norm(loc=m_i, scale=1).pdf(x)
                for m_i, ϕ_i in zip(self.m.squeeze(), self.ϕ.mean(0).squeeze())
            ]
        ).sum(0)
```

The following plot illustrates a fit of the model to simulated data with $N=100$, $\mu=[-4, 0, 9]$ and equal mixture component probabilities.

![CAVI Gaussian mixture model fit.](/images/variational-inference/fit.png)

[^1]: Note that I have slightly altered the notation of the paper using $\mathbf{z}$ instead of $\mathbf{c}$ and $n$ instead of $i$.
[^2]: $\operatorname{Var}(X)=\mathbb{E}[X^2]-\mathbb{E}[X]^2$
