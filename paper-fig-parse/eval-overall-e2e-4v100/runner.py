import os, sys, copy
sys.path.append(os.getcwd()+'/../common')
from runner_helper import Model, Dataset, CachePolicy, ConfigList, percent_gen
from runner_helper import percent_gen as pg

do_mock = False
durable_log = True
fail_only = False

cur_common_base = (ConfigList()
  .override('root_path', ['/nvme/songxiaoniu/graph-learning/wholegraph'])
  .override('logdir', ['run-logs',])
  .override('num_workers', [4])
  .override('epoch', [4])
  .override('skip_epoch', [2])
  .override('presc_epoch', [2])
  .override('use_amp', [True])
  .override('empty_feat', [24])
  .override('omp_thread_num', [40])
  )

cfg_list_collector = ConfigList.Empty()
def lastof(l:list):
  return l[-1:]

'''
GraphSage
'''
# 1.1 unsup, large batch
cur_common_base = (cur_common_base.copy().override('model', [Model.sage]).override('unsupervised', [True]))
cur_common_base = (cur_common_base.copy().override('batchsize', [2000]).override('local_step', [250]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M_undir, ]).override('cache_percent', lastof(pg(1,2,1) + pg(4,8,4) + [0.10])))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.friendster,       ]).override('cache_percent', lastof(pg(1,2,1) + pg(4,6,2) + [0.08])))
# # # 1.2 unsup, mag 240 requires different batch
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.mag240m_homo,     ]).override('cache_percent', lastof([0.002, 0.006, 0.01])).override('batchsize', [ 500]))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.mag240m_homo,     ]).override('cache_percent', lastof([0.002, 0.006, 0.01])).override('batchsize', [1000]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.mag240m_homo,     ]).override('cache_percent', lastof([0.002, 0.006, 0.01])).override('batchsize', [2000]))


# 1.1 sup, large batch
cur_common_base = (cur_common_base.copy().override('unsupervised', [False]))
cur_common_base = (cur_common_base.copy().override('batchsize', [8000]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M_undir, ]).override('cache_percent', lastof(pg(1,2,1) + pg(4,8,4) + [0.11])))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.friendster,       ]).override('cache_percent', lastof(pg(1,2,1) + pg(4,6,2) + [0.08])))
# # # 1.2 sup, mag 240 requires different batch
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.mag240m_homo,     ]).override('cache_percent', lastof([0.002, 0.006, 0.01])).override('batchsize', [4000]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.mag240m_homo,     ]).override('cache_percent', lastof([0.002, 0.006, 0.01])).override('batchsize', [8000]))


cfg_list_collector.hyper_override(
  ['use_collcache', 'cache_policy', "coll_cache_no_group", "coll_cache_concurrent_link"], 
  [
    [True, CachePolicy.clique_part, "DIRECT", ""],
    [True, CachePolicy.clique_part, "", "MPSPhase"],
    [True, CachePolicy.rep, "DIRECT", ""],
    [True, CachePolicy.rep, "", "MPSPhase"],
    [True, CachePolicy.coll_cache_asymm_link, "", "MPSPhase"],
    [True, CachePolicy.coll_cache_asymm_link, "DIRECT", ""],
    [False, CachePolicy.coll_cache_asymm_link, "", ""]
])
cfg_list_collector.override('coll_cache_scale', [
  # 0,
  16,
])

if __name__ == '__main__':
  from sys import argv
  for arg in argv[1:]:
    if arg == '-m' or arg == '--mock':
      do_mock = True
    elif arg == '-i' or arg == '--interactive':
      durable_log = False
    elif arg == '-f' or arg == '--fail':
      fail_only = True
  cfg_list_collector.run(do_mock, durable_log, fail_only=fail_only)