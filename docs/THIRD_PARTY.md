# Third-party algorithm references

FedSApro is an independent experiment harness. The implementations follow the published
algorithms below and do not vendor model weights or datasets.

- IndexGuard: Javad Dogani, Devris Isler, Nikolaos Laoutaris, ICML 2026. The Top-K salience,
  index-overlap similarity, average-linkage clustering, cohesion margin, and minority-size test
  are adapted from the authors' MIT-licensed reference implementation at
  <https://github.com/JavadDogani/IndexGuard>, commit
  `2bd7552f4982b4e99e7a32739fe367e81b896421`.
- FLAME: Nguyen et al., USENIX Security 2022.
- FLTrust: Cao et al., NDSS 2021.
- Safe-FedLLM: Tao et al., *Safe-FedLLM: Delving into the Safety of Federated
  Large Language Models*, ACL 2026, <https://aclanthology.org/2026.acl-long.1120/>.
  The Shadow-Level weighting logic and LoRA-probe structure were checked against the authors'
  Apache-2.0 reference implementation at <https://github.com/dmqx/Safe-FedLLM>, commit
  `0722f3f5945e907b70e53413732d4dd735a25aec`. FedSApro's terminal-delta adaptation is documented
  in `METHODOLOGY.md` and is not claimed to be a verbatim port.
- FedGraph: Chen et al., *FedGraph: Defending Federated Large Language Model Fine-Tuning Against
  Backdoor Attacks via Graph-Based Aggregation*, arXiv:2610.18430, accepted at the ICLR 2026
  Workshop on Principled Design for Trustworthy AI,
  <https://arxiv.org/abs/2610.18430>. The cosine graph, three centralities, and HDBSCAN rule follow
  the paper; the calibrated LoRA slice and threshold are disclosed in `METHODOLOGY.md`.

SCAFFOLD (Karimireddy et al., ICML 2020) and FedProx (Li et al., MLSys 2020) remain implemented
only for legacy profiles and are not included in the current security-study comparison.

