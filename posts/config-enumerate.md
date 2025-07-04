---
title: "Config Enumerate in Pyro"
subtitle: "Understanding Pyro’s enumeration strategy for discrete latent variables."
date: "2020-11-19"
---

Let us consider a standard text book problem (this one is in fact from David Mac Keys superb Information theory, Inderence and Learning Algorithms book): consider that we blindly draw a urn from set of ten urns each containing $10$ balls. Urn $u$ contains $u$ black balls and $$10-u$$ white balls, and we draw from our chosen urn $N$ times with replacement from that urn, obtaining in this way $nB$ black and $N-nB$ white balls. After drawing from the urn $N=10$ times we ask ourselves which urn we have drawn from.
