---
title: "Deriving the ELBO estimator from a Maximum Likelihood perspective"
subtitle: "In this article, we take closer look at the ELBO estimator and derive in the context of variational autoencoders from a maximum likelihood perspective."
date: "2025-01-26"
published: false
---

The few times I came across the ELBO estimator it was derived using KL-divergence. In this article, we take closer look at the ELBO estimator and derive in the context of variational autoencoders from a maximum likelihood perspective.

### Latent variable models

<!-- Complex data types such as images contain a lot of variability. Consider for example images of (human) faces. Faces contain a multitude of features including eyes, noses or hair color. Further these features depend on various other covariates such as sex or ethnicity. A fundamental (and quite reasonable) assumption in latent variable models is that once we have access to these variables or features, which are this context also called latent variables, it is much easier to model or generate such data. However, often we unfortunately do not have accesss to such annotations making it difficult to leverage them. -->

Complex data types, such as images, often exhibit a high degree of variability. For instance, consider images of human faces, which include numerous features such as eyes, noses, or hair color. These features are further influenced by covariates like sex or ethnicity. A core assumption in latent variable models is that if we can identify and access these underlying variables—referred to as latent variables in this context—it becomes significantly easier to model or generate such data. However, in many cases, these annotations are not readily available, making it challenging to fully exploit them.

<!-- Latent variable models describe a large family of different approaches that can be used to learn the "hidden" structure of data. These can be roughly categorised in linear and non-linear approaches. The perhaps most well-known method is principal component analysis (PCA) which decomposes high-dimensional data into factor and loading matrices. The factor matrix can be considered as the latent variables of the data, where each factor is representative for a axis of variation in the data and is encoded in the loading matrix. Crucially, the loading weights are interpretable, meaning that PCA can elucidate important features and thus indicate important sources of variability. Factor models are similar to PCA, but relax some of the stringent assumptions which enforce, for example, orthogonality in the lower dimensional basis representation. -->

Latent variable models encompass a broad class of methods aimed at uncovering the "hidden" structure within data. These approaches can generally be divided into linear and non-linear methods. A well-known example of a linear approach is principal component analysis (PCA), which decomposes high-dimensional data into two components: a factor matrix and a loading matrix. The factor matrix represents the latent variables, with each factor capturing a specific axis of variation in the data, while the loading matrix encodes the contribution of each variable to these factors. Notably, the loading weights are interpretable, allowing PCA to highlight key features and reveal major sources of variability. Factor models extend the ideas of PCA by relaxing some of its strict assumptions, such as the requirement for orthogonality in the low-dimensional representation. Autoencoders can be viewed as a non-linear extension of factor models, which use neural networks to encode high-dimensional data into latent variables. However, a key limitation of PCA, factor models and autoencoders is that they are deterministic[^1], meaning they lack the ability to generate new data samples.

<!--
What does it take to model the new samples ? This fundamentally boils down to modelling the data distribution $p_X(x)$ and turns out to be not trivial in the case latent variable models. To understand this, recall that we don't know latent variables beforehand and so this becomes some sort of chicken and egg problem. We know that if we had only access to the latent variables it would be much easier to generate new data points, i.e. if we knew the eycolor, hair color and ethnicity we could generate a new face. On the other hand we cannot evaluate the probability -->

### The marginal log likelihood

What does it take to model new samples? Fundamentally, this requires modeling the data distribution $p_X(x)$, which is not a trivial task when working with latent variable models. The challenge arises from the fact that we do not know the latent variables beforehand, yet we would need them in order to evaluate $p_X(x)$ to train the model, creating a kind of chicken-and-egg problem. To better understand this problem, let us recall what the objective in maximum likelihood learning actually is

$$
\log \prod_{\mathbf{x} \in \mathcal{D}} p_\theta(\mathbf{x} )=\sum_{\mathbf{x} \in \mathcal{D}} \log p_\theta(\mathbf{x})=\sum_{\mathbf{x} \in \mathcal{D}} \log \int_{\mathbf{z}} p_\theta(\mathbf{x}, \mathbf{z}) d\mathbf{z}.
$$

The last part of the equation make the calculation of the likelihood so difficult. Since we do not have access to the latent variable of each data point $\mathbf{x}$, we need to evaluate the joint probability distribution under each possible $\mathbf{z}$ and sum over these values. In other words, we have to compute marginal log probability. It is easy to see that this calculation can get easily computationally intractable, especially if we are dealing with continuous variables.

### Monte Carlo to our rescue

How can we resolve this unsettling situation ? If we cannot evaluate integral in the equation above, perhaps we can sample (representatitive) values for $\mathbf{z}$ and use them to approximate the $p_\theta(\mathbf{x})$. To see this, note that we can rewrite the equation as an expectation with respect to an uniform distribution over the domain of $\mathbf{z}$

$$
p_\theta(\mathbf{x}) = \underbrace{|\mathcal{Z}| \cdot \frac{1}{|\mathcal{Z}|}}_{1} \int_{\mathbf{z}} p_\theta(\mathbf{x}, \mathbf{z}) d\mathbf{z} = |\mathcal{Z}| \int_{\mathbf{z}}  \frac{1}{|\mathcal{Z}|}  p_\theta(\mathbf{x}, \mathbf{z}) d\mathbf{z}=|\mathcal{Z}|\color{purple}\mathbb{E}_{\mathbf{z}\sim\mathcal{U}(\mathcal{Z})}[p_\theta(\mathbf{z},\mathbf{x})],
$$

where the caligraphic Z denotes the cardinality of $\mathbf{z}$. The idea here is to make the computation cheaper by sampling a few $\mathbf{z}$s instead of assessing each possible completion (Monte Carlo sampling). In other words, we first sample $k$ realisations of $\mathbf{z}$ and approximate the expectation

$$
\color{purple}\mathbb{E}_{\mathbf{z}\sim\mathcal{U}(\mathcal{Z})}[p_\theta(\mathbf{z},\mathbf{x})]\color{black} \approx \frac{1}{k} \sum_{j=1}^{k} p_{\theta}\left(\mathbf{x}, \mathbf{z}^{(j)}\right).
$$

using the sample average[^2]. This, however, doesn't work in practice, simply because there are to many realisations of $\mathbf{z}$ to chose from, making it very unlikely that we chose reasonable samples. In order to better approximate the likelihood $p_\theta(\mathbf{x})$, we need to find ways to reduce the variance of this estimator by sampling $z$'s that are more likely.

### Importance sampling

Let us continue with another attempt. This time, however, we work with an proposal distribution $q_\lambda(z)$ parameterised by a set of parameters $\lambda$ instead of a simple uniform distribution. We start by multiplying $\frac{q_\lambda(z)}{q_\lambda(z)}=1$ into the equation. It is easy to see, that we can then rewrite

$$
p_\theta(\mathbf{x}) = \int_{\mathbf{z}} \frac{q_\lambda(z)}{q_\lambda(z)} p_\theta(\mathbf{x}, \mathbf{z}) d\mathbf{z} =  \int_{\mathbf{z}} q_\lambda(z) \frac{  p_\theta(\mathbf{x}, \mathbf{z})}{q_\lambda(z)}  d\mathbf{z}=\color{green}\mathbb{E}_{\mathbf{z}\sim q_\lambda(z)}[\frac{p_\theta(\mathbf{z},\mathbf{x})}{q_\lambda(z)}],
$$

as an expectation with respect to our proposal distribution. We can then use again Monte carlo sampling

$$
p_\theta(\mathbf{x}) =  \color{green}\mathbb{E}_{\mathbf{z}\sim q_\lambda(z)}[\frac{p_\theta(\mathbf{z},\mathbf{x})}{q_\lambda(z)}] \color{black}\approx \frac{1}{k} \sum_{j=1}^{k} \frac{p_{\theta}\left(\mathbf{x}, \mathbf{z}^{(j)}\right)}{q_\lambda\left(\mathbf{z}^{(j)}\right)}
$$

to obtain a estimate for $p_\theta(\mathbf{x})$. So how does this help with our problem ? Well, it is easy to see if we would have a good proposal distribution $q_\lambda (\mathbf{z})$ under which the sampled realisations of $\mathbf{z}$ are likely, we eventually can obtain a better (less variable) estimate of the marginal density.

### Getting the ELBO

We now have almost all ingredients for deriving the ELBO estimator. However, remember that we actually are interested in the logarithm of the marginal density $\log{p_\theta(\mathbf{x})}$. A naive approach is to take the log of the Monte Carlo estimate above

$$
\log{p_\theta(\mathbf{x})} =  \log \color{green}\mathbb{E}_{\mathbf{z}\sim q_\lambda(z)}[\frac{p_\theta(\mathbf{z},\mathbf{x})}{q_\lambda(z)}] \color{black}\approx \log \frac{1}{k} \sum_{j=1}^{k} \frac{p_{\theta}\left(\mathbf{x}, \mathbf{z}^{(j)}\right)}{q_\lambda\left(\mathbf{z}^{(j)}\right)}
$$

This comes with a problem however. To make this more clear we consider the a single example, i.e.

$$
\log{p_\theta(\mathbf{x})} =  \log \color{green}\mathbb{E}_{\mathbf{z}^{(1)}\sim q_\lambda(z)}[\frac{p_\theta(\mathbf{z}^{(1)},\mathbf{x})}{q_\lambda(z)}] \color{black}\approx \log  \frac{p_{\theta}\left(\mathbf{x}, \mathbf{z}^{(1)}\right)}{q_\lambda\left(\mathbf{z}^{(1)}\right)}
$$

- PCA
- Autoencoder (determnisitic)
- Variational Autoencoder

[^1]: There are probabilistic versions of PCA and factor models that can in principle regarded as generative.
[^2]:
    By the Law of Large Numbers.
    Let's go deeper into **point 2 (Implications for Likelihood Estimation)** and **point 3 (Monte Carlo Approximation Error)** with an example.
