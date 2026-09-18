"""Seed sweep driver: one (agent, seed) training run per process, several in parallel.

No model, environment or training code lives here. The driver executes the definition
cells of single_edge_rl.ipynb (Environment, DQNFlex, the agents, replay, train(),
the data cell) into a namespace and calls them, so a run is the notebook's code. Edit the
notebook and the driver follows. The small sweep helper that seeds, builds the environment,
trains and evaluates lives at the top of this file.

Agents
  WC      DQNAgent_WC       joint recommendation and cache admission
  NC      DQNAgent_NC       recommendation-only DQN, popularity caching rule
  NCPBRS  DQNAgent_NC_PBRS  NC with WC's potential-based reward shaping (control)

Protocol (set in the notebook's config cell, written to results/<out>/seed_sweep_meta.json):
MovieLens, N=5000, K=10, corr_threshold=0.6, prob_leave=0.05, 20 000 training and 10 000
evaluation episodes, best checkpoint, metric = total hits / total steps over the
evaluation episodes with the hit counted against the cache before the action. Within a
seed every agent gets the same environment and the same evaluation episodes.

Usage, from the single-edge-rl-joint-rec-caching folder
  python run_seed_sweep.py --smoke                            # wiring check, 3 agents, seed 0
  python run_seed_sweep.py --launch --jobs WC:0-9,NC:0-9      # headline table
  python run_seed_sweep.py --launch --jobs NCPBRS:0-9         # control
  python run_seed_sweep.py --launch --jobs WC:7 --parallel 1  # a single run
  python run_seed_sweep.py --collect                          # tables, paired tests, comparison with shipped

Output, one file per run so parallel processes never write the same file; finished runs
are skipped when launching again
  results/<out>/seed_sweep_meta.json      protocol, notebook md5, software versions
  results/<out>/runs/<tag>.csv            one row per run: metrics, protocol, minutes, peak RSS
  results/<out>/runs/<tag>_history.npz    per-episode training hit rate and loss
  results/<out>/logs/<tag>.log            stdout/stderr of the run
  results/<out>/all_runs.csv, seed_sweep_results.csv, seed_sweep_aggregate.json   (--collect)
"""
import argparse
import csv
import functools
import glob
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

NB_PATH = 'single_edge_rl.ipynb'
RESULTS_ROOT = 'results'
DEFAULT_OUT = 'new_sweep'
SHIPPED = {  # shipped results, shown next to new runs by --collect
    'headline': os.path.join(RESULTS_ROOT, 'headline_WC_vs_NC', 'seed_sweep_results.csv'),
    'control': os.path.join(RESULTS_ROOT, 'control_NC_PBRS', 'seed_sweep_results.csv'),
}
U_MEAN_OF_RECORD = 0.08565857   # mean of the u matrix the shipped results used (MovieLens 25M, N=5000)
AGENTS = {'WC': 'DQNAgent_WC', 'NC': 'DQNAgent_NC', 'NCPBRS': 'DQNAgent_NC_PBRS'}
DEFAULT_JOBS = 'WC:0-9,NC:0-9'


# Sweep helpers, used when the notebook does not define them itself. One fresh Environment
# before training and one before evaluation, so every agent of a seed starts from the same
# state and is evaluated on the same episodes.
SWEEP_HELPERS = '''
def _build_env():
    return Environment(number_of_contents, cache_size, rewards, number_of_recommendations,
                       corr_threshold, 1 - gamma, user_type="quality_aware",
                       popularity=popularity, u=u)


def _run_rl_agent(agent_cls, seed):
    set_seed(seed)
    env = _build_env()
    agent = agent_cls(env, state_features=7, learning_rate=1e-4, gamma=gamma,
                      eps=eps_value, max_iter=SWEEP_TRAIN_ITERS, replay_type="PER",
                      batch_size=32, replay_buffer_size=10_000, taf=0.005)
    _, _, costs, train_curve = train(env, agent, None, max_iter=SWEEP_TRAIN_ITERS,
                                     plot_interval=10**9, eps_decay=eps_decay,
                                     threshold=converge_threshold, path="",
                                     train_enable=True, save_load=False)
    if getattr(agent, "best_state_dict", None) is not None:
        agent.q_network_policy.load_state_dict(agent.best_state_dict)
    set_seed(seed)
    env = _build_env()
    agent.env = env
    train(env, agent, None, max_iter=SWEEP_EVAL_ITERS, plot_interval=10**9,
          eps_decay=eps_decay, threshold=converge_threshold, path="",
          train_enable=False, save_load=False)
    micro, macro = agent.last_eval_micro, agent.last_eval_macro
    train_curve = np.asarray(train_curve, dtype=np.float32)
    loss_curve  = np.asarray(costs, dtype=np.float32)
    del agent, env; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return micro, macro, train_curve, loss_curve
'''


def run_tag(agent, seed, ns):
    return f"SE_ML_N{ns['number_of_contents']}_K{ns['cache_size']}_{agent}_s{seed}"


def md5(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def build_namespace(train_iters, eval_iters):
    """Execute the notebook's definition cells, the config cells (dataset switch, u and
    popularity load-or-build, Environment) and the sweep helper cell."""
    os.environ.setdefault('MPLBACKEND', 'Agg')
    with open(NB_PATH, encoding='utf-8') as f:
        nb = json.load(f)
    code = [(i, ''.join(c['source'])) for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code']
    i_cfg = next(i for i, s in code if s.startswith('list_of_results=[]'))
    i_eng = next((i for i, s in code if s.startswith('def _build_env():')), None)
    ns = {'__name__': '__seed_sweep__'}
    for i, s in code:
        if not s.strip() or (i_eng is not None and i > i_eng):
            continue
        if i > i_cfg and i != i_eng and not s.startswith('number_of_contents='):
            continue                      # run/plot cells after the config are not executed
        exec(compile(s, f'<nb cell {i}>', 'exec'), ns)
        if 'from tqdm.notebook import' in s:  # plain tqdm, one line per minute
            import tqdm as _tqdm
            ns['trange'] = functools.partial(_tqdm.trange, mininterval=60, file=sys.stderr)
            ns['tqdm'] = functools.partial(_tqdm.tqdm, mininterval=60, file=sys.stderr)
        if s.startswith('number_of_contents='):
            ns['SWEEP_TRAIN_ITERS'], ns['SWEEP_EVAL_ITERS'] = train_iters, eval_iters
            ns['gc'] = __import__('gc')
    if i_eng is None:                     # notebook without sweep cells: define the helpers here
        exec(compile(SWEEP_HELPERS, '<sweep helpers>', 'exec'), ns)
    # sanity: the control has the shaping block, plain NC does not
    cls_src = {s.split('(')[0].split()[-1]: s for _, s in code if s.startswith('class DQNAgent_')}
    assert 'shaping_reward' in cls_src['DQNAgent_NC_PBRS'], 'DQNAgent_NC_PBRS lost its shaping block'
    assert 'shaping_reward' not in cls_src['DQNAgent_NC'], 'DQNAgent_NC must be the plain (unshaped) baseline'
    for req in ('Environment', 'train', 'set_seed', '_run_rl_agent', *AGENTS.values()):
        if req not in ns:
            raise RuntimeError(f'{req} not defined by the notebook')
    um = float(ns['u'].mean())
    if abs(um - U_MEAN_OF_RECORD) > 1e-6:
        print(f'NOTE: u.mean()={um:.8f} differs from the shipped data ({U_MEAN_OF_RECORD}); '
              'different MovieLens release or N; numbers will not match the shipped tables exactly.',
              flush=True)
    return ns


def experiment_def(ns, train_iters, eval_iters, smoke):
    import torch
    return {
        'EXPERIMENT_ID': 'single_edge_seed_sweep',
        'dataset_kind': ns['DATASET_KIND'], 'u_mean': float(ns['u'].mean()), 'u_mean_of_record': U_MEAN_OF_RECORD,
        'number_of_contents': ns['number_of_contents'], 'cache_size': ns['cache_size'],
        'corr_threshold': ns['corr_threshold'], 'prob_leave': ns['prob_to_leave'], 'gamma': ns['gamma'],
        'code_snapshot': {'notebook': NB_PATH, 'md5': md5(NB_PATH), 'driver_md5': md5(__file__)},
        'software': {'python': platform.python_version(), 'torch': torch.__version__,
                     'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
                     'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        'environment_version': 'single-edge, tuple mode, persistent cache (cache carries over episodes)',
        'state_representation_version': 'state_features=7 (continuous u as 7th feature)',
        'action_space_version': {'WC': 'M x (K+1) joint (recommend, admit-into-slot)',
                                 'NC/NCPBRS': 'M rec-only + observed_pop caching heuristic'},
        'reward_version': {'NC': 'hit=1/miss=0',
                           'WC': 'hit=1/miss=0 + PBRS Phi=secured+(1-secured)*fallback, scale 0.1, inside learn()',
                           'NCPBRS': 'same as WC (identical Phi/scale), rec-only action space'},
        'scheduler_type': 'OneCycleLR(max_lr=lr, pct_start=0.3), stepped per episode',
        'optimizer_config': 'AdamW; sigma group lr=5e-5 wd=0; rest lr=1e-4 wd=1e-3; grad clip 5.0',
        'target_sync': 'hard copy every 500 episodes',
        'checkpoint_selection_metric': '1000-episode rolling mean of the per-episode cache-hit rate, second half of '
                                       'training (train() ckpt_window); best_state_dict loaded before eval',
        'metric_definitions': {
            'cache_hit_MICRO': {'computation_time': 'BEFORE_ACTION',
                                'definition': 'total hits / total steps, pooled over eval episodes',
                                'role': 'METRIC OF RECORD'},
            'cache_hit_MACRO': {'computation_time': 'BEFORE_ACTION',
                                'definition': 'mean of per-episode hit/steps', 'role': 'diagnostic only'}},
        'train_iters': train_iters, 'eval_iters': eval_iters,
        'seeding_protocol': 'set_seed(seed) + fresh Environment before train and again before eval '
                            '(notebook _run_rl_agent) -> identical eval episodes across agents within a seed',
        'execution': 'one (agent, seed) per OS process, N processes concurrently on one GPU',
        'smoke': smoke, 'created': time.strftime('%Y-%m-%d %H:%M:%S'),
    }


# ----------------------------------------------------------------------------- one run
def run_one(a):
    out = os.path.join(RESULTS_ROOT, a.out)
    for d in ('runs', 'logs'):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    t0 = time.time()
    ns = build_namespace(a.train_iters, a.eval_iters)
    tag = run_tag(a.agent, a.seed, ns)
    row_path = os.path.join(out, 'runs', f'{tag}.csv')
    if os.path.isfile(row_path) and not a.force:
        print(f'{tag}: row exists, skipping'); return
    meta_path = os.path.join(out, 'seed_sweep_meta.json')
    if not os.path.isfile(meta_path):
        with open(meta_path, 'w') as f:
            json.dump(experiment_def(ns, a.train_iters, a.eval_iters, a.smoke), f, indent=2)
    print(f'===== {tag}: train {a.train_iters} / eval {a.eval_iters} =====', flush=True)
    micro, macro, train_curve, loss_curve = ns['_run_rl_agent'](ns[AGENTS[a.agent]], a.seed)
    mins = (time.time() - t0) / 60.0
    try:
        import psutil
        peak_mb = psutil.Process().memory_info().peak_wset / 2**20
    except Exception:
        peak_mb = float('nan')
    import torch
    row = dict(agent=a.agent, seed=a.seed, micro=float(micro), macro=float(macro),
               dataset=ns['DATASET_KIND'], N=ns['number_of_contents'], K=ns['cache_size'],
               corr_threshold=ns['corr_threshold'], train_iters=a.train_iters, eval_iters=a.eval_iters,
               minutes=round(mins, 1), peak_rss_mb=round(peak_mb), notebook_md5=md5(NB_PATH),
               torch=torch.__version__, finished=time.strftime('%Y-%m-%d %H:%M:%S'), smoke=a.smoke)
    with open(row_path, 'w', newline='') as f:   # one file per run
        w = csv.DictWriter(f, fieldnames=list(row)); w.writeheader(); w.writerow(row)
    ns['np'].savez_compressed(os.path.join(out, 'runs', f'{tag}_history.npz'),
                              train_hit_curve=ns['np'].asarray(train_curve), loss_curve=ns['np'].asarray(loss_curve))
    print(f'[{tag}] MICRO={micro:.4f} MACRO={macro:.4f}  ({mins:.1f} min, peak RSS {peak_mb:.0f} MB) -> {row_path}', flush=True)


# ----------------------------------------------------------------------------- launcher
def parse_jobs(spec):
    jobs = []
    for part in spec.split(','):
        agent, seeds = part.split(':')
        agent = agent.strip().upper()
        if agent not in AGENTS:
            sys.exit(f'unknown agent {agent!r}; choose from {sorted(AGENTS)}')
        if '-' in seeds:
            lo, hi = seeds.split('-'); seeds = range(int(lo), int(hi) + 1)
        else:
            seeds = [int(s) for s in seeds.split('+')]
        jobs += [(agent, int(s)) for s in seeds]
    return jobs


def launch(a):
    out = os.path.join(RESULTS_ROOT, a.out)
    os.makedirs(os.path.join(out, 'logs'), exist_ok=True)
    done = {os.path.basename(p)[:-4] for p in glob.glob(os.path.join(out, 'runs', '*.csv'))}
    todo = [(ag, s) for ag, s in parse_jobs(a.jobs)
            if a.force or not any(d.endswith(f'_{ag}_s{s}') for d in done)]
    print(f'jobs to run: {todo}  (parallel={a.parallel}, out={out})', flush=True)
    env = dict(os.environ, OMP_NUM_THREADS=str(max(1, (os.cpu_count() or 4) // max(1, a.parallel))),
               PYTHONIOENCODING='utf-8', MPLBACKEND='Agg')
    running = {}
    t0 = time.time()
    while todo or running:
        while todo and len(running) < a.parallel:
            ag, s = todo.pop(0); key = f'{ag}_s{s}'
            log = open(os.path.join(out, 'logs', f'{key}.log'), 'a', encoding='utf-8')
            log.write(f'\n===== launched {time.strftime("%Y-%m-%d %H:%M:%S")} =====\n'); log.flush()
            cmd = [sys.executable, os.path.abspath(__file__), '--agent', ag, '--seed', str(s), '--out', a.out,
                   '--train-iters', str(a.train_iters), '--eval-iters', str(a.eval_iters)]
            if a.smoke: cmd.append('--smoke')
            if a.force: cmd.append('--force')
            p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
            running[key] = (p, log)
            print(f'[{(time.time()-t0)/60:6.1f} min] started {key} (pid {p.pid})', flush=True)
            time.sleep(a.stagger)
        time.sleep(10)
        for key, (p, log) in list(running.items()):
            if p.poll() is not None:
                log.close(); del running[key]
                print(f'[{(time.time()-t0)/60:6.1f} min] {key} exited rc={p.returncode}', flush=True)
    print(f'all done in {(time.time()-t0)/3600:.2f} h', flush=True)
    collect(a)


# ----------------------------------------------------------------------------- collect
def collect(a):
    import numpy as np
    out = os.path.join(RESULTS_ROOT, a.out)
    rows = []
    for p in sorted(glob.glob(os.path.join(out, 'runs', '*.csv'))):
        with open(p, newline='') as f:
            rows.extend(csv.DictReader(f))
    if not rows:
        print('no run rows yet in', out); return
    with open(os.path.join(out, 'all_runs.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    new = {(r['agent'], int(r['seed'])): float(r['micro']) for r in rows}
    seeds = sorted({s for _, s in new})
    ref = {}
    for name, path in SHIPPED.items():
        if os.path.isfile(path):
            with open(path, newline='') as f:
                for r in csv.DictReader(f):
                    d = ref.setdefault(int(r['seed']), {})
                    for k in ('WC_micro', 'NC_micro', 'NCPBRS_micro'):
                        if k in r and r[k] not in ('', 'nan'):
                            d.setdefault(k, float(r[k]))
    fields = ['seed'] + [f'{ag}_micro' for ag in AGENTS] + [f'shipped_{ag}_micro' for ag in AGENTS]
    wide = []
    for s in seeds:
        r = {'seed': s}
        for ag in AGENTS:
            r[f'{ag}_micro'] = new.get((ag, s), float('nan'))
            r[f'shipped_{ag}_micro'] = ref.get(s, {}).get(f'{ag}_micro', float('nan'))
        wide.append(r)
    with open(os.path.join(out, 'seed_sweep_results.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(wide)
    agg = {'n_seeds': len(seeds), 'seeds': seeds}
    for ag in AGENTS:
        v = np.array([r[f'{ag}_micro'] for r in wide]); v = v[~np.isnan(v)]
        if v.size:
            agg[f'{ag}_micro_mean'] = float(v.mean()); agg[f'{ag}_micro_std'] = float(v.std(ddof=1)) if v.size > 1 else None
    for base in ('NC', 'NCPBRS'):
        d = np.array([r['WC_micro'] - r[f'{base}_micro'] for r in wide])
        d = d[~np.isnan(d)]
        if d.size:
            e = {'n': int(d.size), 'mean': float(d.mean()), 'std': float(d.std(ddof=1)) if d.size > 1 else None,
                 'wc_wins': int((d > 0).sum())}
            if d.size > 2:
                from scipy import stats
                e['paired_t_p'] = float(stats.ttest_rel(d, np.zeros_like(d)).pvalue)
                try:
                    e['wilcoxon_p'] = float(stats.wilcoxon(d).pvalue)
                except ValueError:
                    e['wilcoxon_p'] = None
            agg[f'WC_minus_{base}'] = e
    with open(os.path.join(out, 'seed_sweep_aggregate.json'), 'w') as f:
        json.dump(agg, f, indent=2)
    hdr = f"{'seed':>4} | " + ' '.join(f'{ag:>7}' for ag in AGENTS) + ' | shipped: ' + ' '.join(f'{ag:>7}' for ag in AGENTS)
    print(hdr)
    for r in wide:
        print(f"{r['seed']:>4} | " + ' '.join(f"{r[f'{ag}_micro']:7.4f}" for ag in AGENTS) + ' |          '
              + ' '.join(f"{r[f'shipped_{ag}_micro']:7.4f}" for ag in AGENTS))
    print(json.dumps(agg, indent=2))
    print('->', os.path.join(out, 'seed_sweep_results.csv'))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--agent', default='WC', help='WC | NC | NCPBRS (single-run mode)')
    ap.add_argument('--seed', type=int)
    ap.add_argument('--launch', action='store_true')
    ap.add_argument('--jobs', default=DEFAULT_JOBS, help='e.g. WC:0-9,NC:0-9  or  NCPBRS:3+5')
    ap.add_argument('--collect', action='store_true')
    ap.add_argument('--smoke', action='store_true', help='250/60 iters, all agents seed 0, results/smoke/')
    ap.add_argument('--parallel', type=int, default=2, help='concurrent runs, about 2 GB RAM each')
    ap.add_argument('--stagger', type=float, default=45, help='seconds between process starts')
    ap.add_argument('--train-iters', type=int, default=20000)
    ap.add_argument('--eval-iters', type=int, default=10000)
    ap.add_argument('--out', default=None, help=f'sub-folder of {RESULTS_ROOT}/ (default {DEFAULT_OUT})')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    a.agent = a.agent.upper()
    if a.smoke:
        a.train_iters = min(a.train_iters, 250); a.eval_iters = min(a.eval_iters, 60)
        a.out = a.out or 'smoke'
        if a.jobs == DEFAULT_JOBS: a.jobs = 'WC:0,NC:0,NCPBRS:0'
        if a.seed is None and not a.collect: a.launch = True
    a.out = a.out or DEFAULT_OUT
    if a.collect:
        collect(a)
    elif a.launch:
        launch(a)
    elif a.seed is not None:
        run_one(a)
    else:
        ap.print_help()


if __name__ == '__main__':
    main()
