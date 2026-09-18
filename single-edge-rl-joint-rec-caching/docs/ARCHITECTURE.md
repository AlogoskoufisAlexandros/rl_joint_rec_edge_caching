# How it works

A walk through `single_edge_rl.ipynb`, in the order the cells appear. For the
numbers see [RESULTS.md](RESULTS.md); for running things see [REPRODUCE.md](REPRODUCE.md).

## 1. Problem

At each step of a user session the agent recommends the next item and decides whether
the item the user just watched goes into an edge cache of fixed size K (evicting one of
the cached items, or nothing). The user either follows the recommendation or picks
something else, then either continues or leaves. Reward is 1 when the item the user
watches next is already in the cache. Both agents are Double DQNs (online network selects
the next action, target network evaluates it; section 7) with dueling NoisyNet heads,
3-step returns and prioritised replay. `DQNAgent_WC` learns both decisions; `DQNAgent_NC`
learns only the recommendation and caches by a popularity rule.

## 2. Environment

`Environment` with `tabular=False`. A state is `(content_id, cache)`, where `cache` is a
list of K item ids. `simulate(action, state)` does one user step:

- `done = random() < prob_leave`. With `prob_leave = 0.05` a session lasts about 20 steps.
- User model `quality_aware`, used for all results: the user follows the recommendation
  if `u[current, recommended] > prob_follow` (which is set to `corr_threshold`, 0.6 on
  MovieLens). Otherwise the next item is drawn from the popularity distribution with the
  current item excluded. There is also a `random` user that follows with a fixed
  probability; it is not used here.
- A session starts at a uniformly random item. The cache is kept from the previous
  session (it is only random at the very first session of a run).
- `u` is the N x N item similarity matrix, diagonal zero. `popularity` is the empirical
  rating-count share of the N items (a shuffled Zipf law for synthetic data).

Hit timing: a hit is counted when the *next* item is in the cache *as it was before the
action*. An admission therefore pays off from the following step on.

`DATASET_KIND` in the config cell selects `'movielens'` (`ratings.csv`, N = 5000,
`corr_threshold` 0.6), `'kuairec'` (`kuairec_small_matrix.csv`, N = 3000, 0.20) or
`'synthetic'` (no file, N = 5000, 0.8). The results are on MovieLens.

## 3. State representation

The network does not see `(content, cache)` directly. At each step `find_topN` picks M
candidate items and `representation_transformation` builds a matrix with one row of F
features per candidate. M is the size of the menu the agent recommends from; the network's
outputs are indexed by these rows (M actions for NC, M x (K + 1) for WC).

M is derived from the data, not set by hand:

```
M = max over items i of |{ j : u[i, j] > prob_follow }|  +  K
```

that is, the largest number of followable recommendations any item has, plus the cache
size (capped at N). On MovieLens with `prob_follow` 0.6 this is 9 + 10 = 19. The bound
guarantees that every item the user could follow from the current one, and every cached
item, always fits in the menu.

The rows are filled in a fixed order: row 0 is the current item, rows 1..K are the cached
items in cache order, and the remaining rows are the items most similar to the current one
(from a precomputed argsort of `u`), skipping any already listed. The row order matters:
`DQNFlex` slices rows 1..K for its cache head, so changing it breaks the cache advantage
and the interaction term without any error. Because the candidate set is rebuilt at every
state, each stored transition also keeps its own item-to-row map (`topN_map`) so that a
replayed action can be mapped back to the row it occupied.

The results use the 7-feature version: is-current, followable-from-current,
is-cached, observed popularity, future cache value, next-step flag, and the similarity
`u[current, candidate]`. The older 5- and 6-feature versions are still in the notebook.

## 4. Actions

- WC: one index over M x (K + 1), decoded as (item to recommend, cache slot that the
  just-watched item replaces). Slot K means "do not cache". Masking removes every action
  that recommends the current item, and, when the current item is already in the cache,
  every action that would admit it again, leaving only the "do not cache" slot for each
  recommendation.
- NC: M actions, the recommendation only. The cache rule: if the current item is not
  cached and its observed popularity is higher than that of the least popular cached
  item, it replaces that item. `observed_pop` is a running count of everything watched
  in the run.

## 5. Reward shaping

Both agents get the base reward, 1 for a hit and 0 otherwise. WC adds a potential-based
shaping term inside `learn()`; the environment reward is unchanged.

```
secured(s)  = 1 if some cached item is followable from the current item, else 0
fallback(s) = share of observed popularity held by the cache
Phi(s)      = secured + (1 - secured) * fallback

F = 0.1 * ( gamma^n * Phi(s') * (1 - done) - Phi(s) )        n = n_step = 3
```

`s'` is the state n steps ahead, matching the n-step return in the replay buffer.
Because F is a difference of a state-only potential, it does not change which policy is
optimal (Ng, Harada and Russell, 1999). One detail: `observed_pop` keeps growing during
training and both potentials are evaluated with the counts at learning time, so the
potential is time-varying. Within one update it is still a proper difference of
potentials.

`DQNAgent_NC_PBRS` is NC with exactly this term added in `learn()`. Its purpose is to
separate the effect of the denser reward from the effect of the larger action space.

## 6. Network

`DQNFlex` takes the [B, M, F] candidate matrix.

1. A gated linear unit encodes each candidate row to 128 dimensions.
2. Cross-attention with the current item (row 0) as the query and all M rows as keys and
   values. The attended context is added to the query and layer-normalised; this
   contextual vector is then added to every candidate row, so each row carries the
   current-item context.
3. Dueling heads built from `NoisyLinear` layers (factorised Gaussian noise,
   `sigma_init = 0.5`, noise only in training mode): a state value from the contextual
   vector, a recommendation advantage over the M candidates, a cache advantage over the K
   cached rows plus a "do not cache" slot (represented by the mean of the cached rows),
   and a bilinear interaction between the two. For WC, Q = V + A_rec + A_cache +
   A_int over M x (K + 1) actions. For NC (`rec_only=True`) the cache and interaction
   heads are dropped and Q = V + A_rec over M actions.

`reset_noise()` is called before each action selection and each learning step.

## 7. Training

Double DQN with 3-step returns and mixed precision. Target:

```
y = r + F + gamma^3 * (1 - done) * Q_target(s', argmax_a Q_online(s', a))
```

The online network is put in eval mode (noise off) while choosing the target action.

- Loss: Huber, weighted by the prioritised-replay importance weights.
- Optimiser: AdamW. Parameters whose name contains `sigma` get no weight decay (decay
  would shrink the exploration noise) and learning rate 5e-5; the rest get weight decay
  1e-3 and learning rate 1e-4. Gradients are clipped at norm 5.
- Learning rate: OneCycleLR stepped once per episode.
- Target network: hard copy every 500 episodes.
- Exploration: NoisyNet only. `eps_value = 0`; the epsilon arguments of the DQN agents
  are unused (only the tabular agents use epsilon-greedy).
- Learning starts once the buffer holds 100 batches. One update per environment step.
- Checkpoint: the weights at the highest 1 000-episode rolling mean of the per-episode hit
  rate, considered in the second half of training, are kept and loaded before evaluation.

## 8. Replay buffer

Prioritised replay on a sum tree: alpha 0.6, beta annealed from 0.4 to 1 over 90% of
training, priority epsilon 0.01, new transitions at maximum priority. Each stored
transition also keeps the candidate list it was built with, because the candidate set is
recomputed per state and the stored action has to be mapped back to its row.

## 9. Agents in the notebook

- Non-RL: `Non_RL_agent_baseline` recommends the most similar item; `Non_RL_agent_greedy`
  recommends the most similar cached item. Both cache by popularity.
- Tabular, for small catalogues: `Agent_PI`, `Agent_without_caching`, `Agent_with_caching`.
  Carried over from the thesis repository; not used in the results.
- Double DQN: `DQNAgent_NC`, `DQNAgent_NC_PBRS`, `DQNAgent_WC`.

## Things to know before changing the code

- NC stops improving after about 1 300 episodes, at a policy that always recommends a
  cached item. That is not a training failure: under this user model an unfollowable
  recommendation costs nothing and a followable cached one is a sure hit, so the policy
  is the best a recommendation-only agent can do. The NoisyNet noise is intact at that
  point (RESULTS.md, section 2). Adding epsilon-greedy exploration would
  not be expected to change NC's number.
- Report the pooled hit rate (total hits / total steps). The mean of per-episode rates is
  several points lower on the same policy because short, cold-start sessions weigh as
  much as long ones.
- The two things most likely to break silently: the candidate row order (section 3) and
  the `(1 - done)` factor in the target (section 7).
