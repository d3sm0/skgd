to check:
	- https://github.com/ClaireLC/backprop_kf_vo
	- https://arxiv.org/pdf/1805.11122.pdf
	- https://arxiv.org/pdf/1511.05121.pdf
	- https://papers.nips.cc/paper/6090-backprop-kf-learning-discriminative-deterministic-state-estimators.pdf
	- https://github.com/tu-rbo/differentiable-particle-filters


call:
- what kind of non-stationarity can it handle?
- do you think can it be made off-policy? if so what kind of IS?
- how can i learn the features in a scalable fashion? (something like proximal operator?) (even in the non linear case the features are not learned or you meant updating all parameter vector?
- where does it break in the stochastic setting?
- what kind of uncertainty is P tracking? does it incorporate uncertainty about the future? or only about present estimates?

Outcome:
- ktd breaks if transitions are not L-smooth (in wasserstein)
- try a different weighting for the data before trying fancy things in pe
- do regression as dqn style if with future value of td
- computational cost is n^2 ... sorry about it
- off policy trpo is possible if no n-step return is used (i.e. no memory)
- use the variance of the ktd to update the vf



Status:
- check correlation between covariance at different lags and change process of the mass of the pole
- different weighting works but not as good as KTD
- instability of target update makes KTD sad

- how to use the variance of estimate of the KTD for the policy?
	- IS doesn't seem to work
	- soft-bellman doesn't seem to work

