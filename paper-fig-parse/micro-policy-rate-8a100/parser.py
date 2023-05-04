import os, sys, math
exp_dir = os.getcwd()+'/../../study/exp-overall-8a100'
sys.path.append(os.getcwd()+'/../../study/common')
sys.path.append(exp_dir)
from common_parser import *
from runner_helper import *
from runner import cfg_list_collector


selected_col = ['short_app']
selected_col += ['policy_impl', 'cache_percentage', 'batchsize']
selected_col += ['dataset_short']
# selected_col += ['num_worker']
# selected_col += ['use_amp', 'epoch_e2e_time', 'cuda_usage']

# selected_col += ['Step(average) L1 sample']
# selected_col += ['Step(average) L1 recv']
# selected_col += ['Step(average) L1 train total']

# selected_col += ['Time.L','Time.R','Time.C']
selected_col += ['Wght.L','Wght.R','Wght.C']
selected_col += ['Step(average) L2 feat copy']
# selected_col += ['Thpt.L','Thpt.R','Thpt.C']
# selected_col += ['SizeGB.L','SizeGB.R','SizeGB.C']
# selected_col += ['optimal_local_rate','optimal_remote_rate','optimal_cpu_rate']
# selected_col += ['coll_cache:local_cache_rate']
# selected_col += ['coll_cache:remote_cache_rate']
# selected_col += ['coll_cache:global_cache_rate']

cfg_list_collector = (cfg_list_collector.copy()
  .override('logdir', [exp_dir + '/run-logs'])
  .select('use_collcache', [True])
  # .select('coll_cache_concurrent_link', ['MPSPhase'])
)

if __name__ == '__main__':
  bench_list = [BenchInstance().init_from_cfg(cfg) for cfg in cfg_list_collector.conf_list]
  # print(bench_list[0].vals)
  with open(f'data.dat', 'w') as f:
    BenchInstance.print_dat(bench_list, f, selected_col)