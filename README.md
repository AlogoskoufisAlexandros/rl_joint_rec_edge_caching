# Joint content recommendation and edge caching with deep RL

Post-graduation follow-on to my diploma thesis
([*Optimizing Network-friendly Recommendations and Caching jointly, using Reinforcement Learning*](https://github.com/AlogoskoufisAlexandros/thesis-network-friendly-recommendations-caching-rl),
TU Crete, 2025). One Double DQN agent decides, at every step of a user session, which item
to recommend next and whether to put the item the user just watched into an edge cache.

| Folder | What it is | Status |
|---|---|---|
| [`single-edge-rl-joint-rec-caching/`](single-edge-rl-joint-rec-caching/) | One edge cache. Redesigned Q-network, 10-seed MovieLens results, seed-sweep driver, docs. | released, `v1.0-single-edge` |
| multi-cache extension | Several cooperating edge caches, built on the single-edge code. | in preparation |

Start with the single-edge folder's [README](single-edge-rl-joint-rec-caching/README.md).
MIT licence (see the folder's `LICENSE`); citation in its `CITATION.cff`.
