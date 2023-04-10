import os, sys, copy
sys.path.append(os.getcwd()+'/../common')
from runner_helper import Framework, Model, Dataset, CachePolicy, ConfigList, percent_gen

do_mock = False
durable_log = True

cur_common_base = (ConfigList()
  .override('root_path', ['/nvme/songxiaoniu/graph-learning/wholegraph'])
  .override('logdir', ['run-logs',])
  .override('num_workers', [8])
  .override('epoch', [4])
  .override('skip_epoch', [2])
  .override('presc_epoch', [1])
  .override('use_amp', [True])
  .override('empty_feat', [26])
  )

cfg_list_collector = ConfigList.Empty()

'''
GraphSage
'''
# 1.1 unsup, large batch
cur_common_base = (cur_common_base.copy().override('model', [Model.sage]).override('unsupervised', [True]))
cur_common_base = (cur_common_base.copy().override('batchsize', [8000]).override('local_step', [125]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M_undir, ]).override('cache_percent', [0.25]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.friendster,       ]).override('cache_percent', [0.25]))
# # 1.2 unsup, mag 240 requires different batch
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.mag240m_homo,     ]).override('cache_percent', [0.08]).override('batchsize', [2000]))


# 1.1 sup, large batch
cur_common_base = (cur_common_base.copy().override('unsupervised', [False]))
cur_common_base = (cur_common_base.copy().override('batchsize', [8000]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M_undir, ]).override('cache_percent', [0.25]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.friendster,       ]).override('cache_percent', [0.25]))
# # 1.2 sup, mag 240 requires different batch
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.mag240m_homo,     ]).override('cache_percent', [0.08]).override('batchsize', [8000]))


cfg_list_collector.hyper_override(
  ['use_collcache', 'cache_policy'], 
  [
    [True, CachePolicy.clique_part],
    [True, CachePolicy.rep],
    [True, CachePolicy.coll_cache_asymm_link],
    [False, CachePolicy.coll_cache_asymm_link]
])

if __name__ == '__main__':
  from sys import argv
  for arg in argv[1:]:
    if arg == '-m' or arg == '--mock':
      do_mock = True
    elif arg == '-i' or arg == '--interactive':
      durable_log = False
  cfg_list_collector.run(do_mock, durable_log)