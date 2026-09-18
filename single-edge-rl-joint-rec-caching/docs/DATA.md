# Data

All files go in `single-edge-rl-joint-rec-caching/`, next to the notebook.

## No download needed to try the code

Set `DATASET_KIND = 'synthetic'` in the notebook's config cell. The simulator then builds
a random similarity matrix and a shuffled Zipf popularity vector from the run seed. Use
this to check that the pipeline runs. The results in RESULTS.md are on MovieLens.

## Datasets

Set `DATASET_KIND` and place the matching file in the root. `normalize_ratings_schema`
detects the column layout.

| `DATASET_KIND` | file | columns | source |
|---|---|---|---|
| `synthetic` | none | | generated in code |
| `movielens` | `ratings.csv` | `userId, movieId, rating, timestamp` | MovieLens 25M, https://grouplens.org/datasets/movielens/25m/ |
| `kuairec` | `kuairec_small_matrix.csv` | `user_id, video_id, watch_ratio` | KuaiRec, from its own distribution page |

Defaults in the config cell: MovieLens N = 5000 and `corr_threshold` = 0.6 (the setting
of the results); KuaiRec N = 3000, 0.20; synthetic N = 5000, 0.8.

## Derived inputs

The notebook, and `run_seed_sweep.py` which executes the same cell, loads
`u_file_<kind>.npy` and `popularity_file_<kind>.npy` if they exist and have the right
size, and otherwise builds them from the raw file (or from the seed for synthetic data)
and saves them. Delete them to force a rebuild. They are git-ignored.

How they are built from MovieLens (`get_top_movies`, `create_u`, `create_popularity`):
the N = 5000 most-rated movies are kept; `popularity` is their share of rating counts;
`u[i, j]` is the cosine similarity of items i and j over item-mean-centred user ratings,
after item-to-item collaborative filtering (k = 10) has filled in missing ratings, clipped
to [0, 1]. The simulated user follows a recommendation of j while watching i if
`u[i, j] > corr_threshold`. The `u` used for the results has mean 0.08565857; at threshold
0.6, 594 of the 5 000 items have at least one followable recommendation.

## Why the raw files are not included

MovieLens is distributed by GroupLens for research and educational use and is not to be
redistributed; KuaiRec has its own terms. Both files are also large (about 680 MB and
390 MB). `.gitignore` excludes them and all `.npy` files.
