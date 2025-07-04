---
title: "Config Enumerate in Pyro"
subtitle: "Understanding Pyro’s enumeration strategy for discrete latent variables."
date: "2020-11-19"
---

Let us consider a standard text book problem (this one is in fact from David Mac Keys superb Information theory, Inderence and Learning Algorithms book): consider that we blindly draw a urn from set of ten urns each containing $10$ balls. Urn $u$ contains $u$ black balls and $10-u$ white balls, and we draw from our chosen urn $N$ times with replacement from that urn, obtaining in this way $nB$ black and $N-nB$ white balls. After drawing from the urn $N=10$ times we ask ourselves which urn we have drawn from.


$$
\begin{aligned}
u &\sim \text{Categorical}([p_0 = 0, p_2 = \frac{1}{10},\dots, p_ {10} = \frac{1}{10} ]) \\
n_B | u &\sim \text{Bin}(n=10, p=\frac{u}{10})
\end{aligned}
$$

The posterior probability distribution is

$$
p(u|n_b) = \frac{p(n_b|u) \cdot p(u)}{p(n_b)}
$$

which we can easily determine analytically, but here we rather use Pyro. However, we first start by defining a function that let us simulate the experiment described above.

```python
import torch
import pyro
import pyro.distributions as dist
import numpy as np
import matplotlib.pyplot as plt

def sample(n=1):
    urn = pyro.distributions.Categorical(torch.ones(10)/10).sample()
    draws = pyro.distributions.Binomial(10, urn/10).sample((n,))
    return urn, draws
```

In the first line we sample uniformly from a Categorical distribution, as the probability for each category $u=0\dots 9$ is $\frac{1}{10}$, and then draw $10$ times from a Binomial distribution with probability $\frac{u}{10}$. The function returns the true urn, whose probability we seek to determine with a Pyro model, and the actual draw(s) (observations). Also the function enables us to specify the number of times we want to perform the experiment ($n$).

Defining statistical models in Pyro requires us to define a models which in some sense “reverse engineers” the stochastic process of interest. For this reason, our model looks very similar to the function we defined to simulate our data, but let us go through the function line by line:

```python
def model(y):
    u = pyro.sample('u', pyro.distributions.Dirichlet(torch.ones(10)))
    with pyro.plate('data', y.shape[0]):
        urn = pyro.sample('urn', pyro.distributions.Categorical(u))
        pyro.sample('obs', pyro.distributions.Binomial(10, urn/10), obs=y)
```

model first defines a distribution for the probability of each urn, which is in this case a Dirichlet distribution, a common prior for the categorical distribution. Setting the concentration of the Dirichlet to vector of ones generates a flat distribution, thus representing a uniform probability for each urn u. The next statement with `pyro.plate(‘…’)` is a so called plate, a context to indicate conditional independence and enable vectorised computations. Within in the this context we sample from a `Categorical(u)` distribution which will return the chosen urn. Finally, the program evaluates the likelihood of the observations y given the urn u (note these are passed via the kwarg obs in the sample statement).

## Optimising the model

We will use stochastic variational inference (SVI) for the inference and set up the code appropriately. To simplify things, we use the AutoDiagonalNormal guide from which sets up a Normal distribution with diagonal covariance for all hidden variables. The `Trace_ELBO` loss enables to compute the ELBO over graph representation of our model, and finally we use an ADAM to perform the optimisation.


```python
guide = pyro.infer.autoguide.AutoDiagonalNormal(model)
loss = pyro.infer.Trace*ELBO(max_plate_nesting=1)
adam = pyro.optim.Adam({'lr': .001})
svi = pyro.infer.SVI(model, guide, adam, loss)
num_steps = 10000
losses = []
for * in tqdm(range(num_steps)):
    loss = svi.step(draws)
    losses.append(loss)
```

However, executing this code fails with a `NotImplementedError: Cannot transform _IntergerInterval constraints` exception, so what went wrong here ? The error is due to the Categorical distribution which only has discrete support. To make the model work we have to explicitly tell Pyro to enumerate out the variables during training the model. Enumerating may occur sequentially or in parallel, with the latter enabling speed ups as it allows to parallelise computations.

The simplest way to enable enumeration is to decorate our model with `@config_enumerate`.

```python
@pyro.infer.config_enumerate
def model(y): ### rest of the model code
```

This tells Pyro to enumerate all discrete variables in the model. Next we need to instruct the guide about the variables we have enumerated out, or in other words for which variables want variational distributions. Here, we have two possibilities, we could either hide the Categorical “urn” distribution or expose all other variables (“`u`”) with `pyro.poutine.block`.

```python
guide = pyro.infer.autoguide.AutoDiagonalNormal(pyro.poutine.block(model, expose=["u"]))
```

Finally, we have to modify the loss function; rather than using `Trace_ELBO`, we use `TraceEnum_ELBO` which allows for enumeration on model graph.


```python
loss = pyro.infer.TraceEnum_ELBO(max_plate_nesting=1)
```

Let now try out the code. We start by performing the hypothetical urn experiment 10 times, i.e. we select a random urn and then draw 10 times 10 balls with replacement while counting the number of black balls.

```python
true_urn, draws = sample(10)
print(f'True urn {true_urn}, sample {draws}')

> True urn 6, sample tensor([8., 3., 6., 5., 6., 6., 7., 6., 9., 7.])
> 
```

After training the model for 1000 iterations we find that the ELBO has converged.

```python
plt.semilogy(losses)
ax = plt.gca()
ax.set(ylabel='ELBO', xlabel='Step')
```

To obtain the posterior distribution u requires us to write some additional line of code,

```python
posterior = pyro.infer.Predictive(model, guide=guide, num_samples=5000)
params = posterior(draws)
posterior_u = params['u'].detach().numpy()
```

which draw from the fitted posterior distributions 5000 samples. To figure out which urn we have most likely drawn from we plot the resulting distribution

```python
plt.bar(np.arange(10), postU.mean(0).reshape(-1))
_ = plt.gca().set(
    xlabel='Urn',
    ylabel='Probability',
    title='Posterior distribution of $u$'
)
```

indicating that urn 6 is the most likely urn.
