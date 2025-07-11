---
title: "Fully Connected Sigmoid Networks"
subtitle: "In this article, I explore how autoregression, typically used in temporal or sequential data, can be applied to images."
date: "2025-01-23"
published: true
---

The context of autoregression in terms of temporal data or language is pretty straightforward. Stock prices can rise or fall over time, similarly the order of words in a sentence is not random or independent of each other. But how can we understand autoregression in the context of images ? One possible way to frame images in an autoregression problem is to look one pixel at a time, starting from the top left corner of the image and moving then to the right until we hit the right border and then to continue in the next line. Then we can regard each individual pixel $x_n$ as a data point in a sequence of $n={1,\dots,N}$ data points. In the case of the Caltech Silhouette dataset (Fig. 1), where the data comprises a set of greyscale images, the individual $x_n$ take either the values 0 or 1.

<figure>
    <img src="/images/fsbvn/image.png" width="400" class="center-image">
    <figcaption>
    Fig. 1: An Aeroplane in the Caltech Silouhette dataset. Modeling images as autoregressive data by viewing consecutive pixels as sequence of data points.
    </figcaption>
</figure>

### Modeling the state space images

Modeling even simple images, such as those from the Caltech Silhouette dataset, is a hard problem. To see this, consider the large state space of such images. Each pixel can take on either the value 0 or 1 and we have 16 $\times$ 16 of them. This gives us in theory $2^{256}$ possible images. In a generative modelling approach, where the goal is to model the data distribtion over $p_X(x)$, this would require us to assign a probability to each individual realisation of this state space. In other words, specifying the joint distribution over such images requires $2^n-1$ parameters (the minus stems that probability distributions have to sum to 1 and have therefore one degree of freedom less). Certainly this is intractable, nor particularly useful because data contain some inherent structure. In fact, most of the random images you could come up with do not make sense at all (Fig. 2).

<figure>
    <img src="/images/fsbvn/random_structured.png" width="600">
    <figcaption>
    Fig. 2: The left-hand side of the figure displays "random" images in which the white and black pixels were sampled independently. The right-hand side of the figure depict actual objects from the Caltech Silhouette dataset.
    </figcaption>
</figure>

<!-- ![Convergence of the model after 10 iterations.](/images/fsbvn/random_structured.png){: .center-image } -->

From the images above it becomes clear that there are dependencies in the image. For example if a particular pixel is white, it is more likely that surrounding pixels are white too and vice versa. For this reason, it is intuitive that modelling the joint distribution of an image $p_X(x_1, \dots,x_n)$ is somewhat wasteful if we assume that each $x_n$ is independent

$$
p_X(x_1,\dots, x_n) = p_X(x_1)p_X(x_2)\dots p_X(x_n)
$$

as we clearly neglect the fact that a particular state of a random variable (pixel) is influenced by the surrounding pixels.

Indeed, one way to address this issue is to incorporate structure into the model. The perhaps simplest structure one can perhaps think of is to make each pixel dependent on the pixels before. This is the simplest form of an autoregressive model. Probabilistically this corresponds to saying

$$
p_X(x_1,\dots, x_n) = p_X(x_1)p_X(x_2|x_1) p_X(x_3|x_1, x_2)\dots p_X(x_n|x_1,\dots, x_{n-1}).
$$

### A better way - imposing an autoregressive structure

In fully visible belief network we parameterize each of the conditionals on the left hand side using simple logistic or sigmoid models

$$
\begin{aligned}
p_{X_1}(X_1=1; \mathbf{W}, \mathbf{b}) &= \sigma(b_{1}) \\
p_{X_2}(X_2=1; x_1, \mathbf{W}, \mathbf{b}) &= \sigma(b_{2} + w_{12} \cdot x_1) \\
&\vdots \\
p_{X_n}(X_n=1; x_1,\dots, x_{n-1}, \mathbf{W}, \mathbf{b}) &= \sigma(b_{n} + w_{1n} \cdot x_1 + w_{2n} \cdot x_2 + \dots +w_{(n-1)n} \cdot x_{n-1})
\end{aligned}
$$

where the $\mathbf{W}$ and $\mathbf{b}$ denote weight and bias parameters of a linear layer in a neural network[^1], and $\sigma$ the sigmoid function. It turns out that implementing such a model is relatively easy. To compute the logits[^2] of the model, we employ simple matrix multiplication but ensure to satisfy the autoregressive structure by masking out some weights in the weight matrix $\mathbf{W}$.

<figure>
    <img src="/images/fsbvn/logits.png" width="600">
    <figcaption>
    Fig. 3: The logit computation in a fully connected sigmoid belief network. The matrix $\mathbf{X}$ denotes a batch of  flattened images (each row depicts a single sample or image), $\mathbf{W}$ a weight matrix in which the white entries are masked out (i.e. they are set to zero) and $\mathbf{b}$ a bias vector.
    </figcaption>
</figure>

Implementing this model is straightforward in `pytorch`. The following code block essentially shows how Fig. 3 is implemented in the forward method.

```python
class FVSBN(nn.Module):
    def __init__(self, data_dim, logits=True):
        super().__init__()
        self.data_dim = data_dim
        self.w = nn.Parameter(torch.randn(data_dim, data_dim))
        self.b = nn.Parameter(torch.randn(1, data_dim))

        self.s = nn.Sigmoid()
        self.l = logits

    def forward(self, x):

        original_shape = x.shape
        x = x.view(original_shape[0], -1)
        # input batch x dim * dim
        alpha = torch.tril(self.w, diagonal=-1)
        logits = x @ alpha.T + self.b
        if self.l:
            return logits.view(original_shape)

        return self.s(logits).view(original_shape)

```

This is essentially all we need to compute all the conditionals per batch. The forward method returns the $B\times N$ matrix of logits. Remember, to train the model we want to maximise the log likelhood of the data. Let $\mathbf{x}^{(j)}$ denote the vector of pixels of an individual image $x_1^{(j)}, \dots, x_N^{(j)}$. Then the log likelihood becomes

$$
\begin{aligned}
\log{p(\mathbf{X})}=\sum_{j=1}^J\sum_{n=1}^N \log{p(x^{(j)}|x^{(j)}_1,\dots,x^{(j)}_{n-1}, \mathbf{W}, \mathbf{b})}
\end{aligned}
$$

and since in this example the pixel values can only take either the values of 0 or 1, we choose a Bernoulli likelihood

$$
\begin{aligned}
x_n^{(j)}&\sim \textrm{Bernoulli}(p_n^{(j)}) \\
p_n^{(j)} &= \sigma(b_n+\sum_k^{n-1}w_{kn}x^{(j)}_k )
\end{aligned}
$$

which can be easily implemented using the `binary_cross_entropy_with_logits` loss function.

```python
def loss_fn(inputs, preds):
    loss = F.binary_cross_entropy_with_logits(preds, inputs, reduction="none").sum()
    return loss / inputs.shape[0]
```

### How to sample from the model

Sampling from autoregressive models is somewhat cumbersome as it requires us to sample each random variable $x_1,\dots,x_n$ sequentially, i.e.

$$
\begin{aligned}
\bar{x}_{1} &\sim p\left(x_{1}\right) \\
\bar{x}_{2} &\sim p\left(x_{2} \mid x_{1}=\bar{x}_{1}\right) \\
\bar{x}_{3} &\sim p\left(x_{3} \mid x_{1}=\bar{x}_{1}, x_{2}=\bar{x}_{2}\right)
\end{aligned}
$$

where $\bar{x_i}$ represent sampled values from the trained model. To enable sampling from our FCSBNs, we attach a sampling method to the model class.

```python
    def sample(self, num_samples=1, device='cuda'):
        sample = torch.zeros(num_samples, self.data_dim).to(device)
        probs = torch.zeros(num_samples, self.data_dim).to(device)
        i = 0
        for i in range(self.data_dim):
            # print(i, sample)
            p_i = F.sigmoid(sample @ torch.tril(self.w, -1).T[:, i] + self.b[:, i])
            probs[:, i] = p_i
            sample[:, i] = torch.bernoulli(p_i)

        return sample, probs
```

### Results from the Caltech Silhoutte dataset

I applied the FCSBN model to the Caltech Silhouette dataset. The following Figures represent some of the sampled images and their corresponding probability maps.

<figure>
    <img src="/images/fsbvn/sampled.png" width="600">
    <img src="/images/fsbvn/probabilities.png" width="600">
    <figcaption>
    Fig. 4: The plot on the top illustrates sampled images from a trained FCSBN, whereas the bottom plot illustrates the corresponding probabilies for each pixel (blue and red denote high and low probabilities respectively).
    </figcaption>
</figure>

[^1]: Strictly speaking it doesn't have to be a neural network, but to think about it as such is useful for implementation purposes.
[^2]: That is the argument for the $\sigma$-function.
