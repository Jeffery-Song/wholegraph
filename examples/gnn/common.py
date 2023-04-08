
kCacheByDegree          = 0
kCacheByHeuristic       = 1
kCacheByPreSample       = 2
kCacheByDegreeHop       = 3
kCacheByPreSampleStatic = 4
kCacheByFakeOptimal     = 5
kDynamicCache           = 6
kCacheByRandom          = 7
kCollCache              = 8
kCollCacheIntuitive     = 9
kPartitionCache         = 10
kPartRepCache           = 11
kRepCache               = 12
kCollCacheAsymmLink     = 13
kCliquePart             = 14
kCliquePartByDegree     = 15

cache_policy_map = {
    'coll_cache'            : kCollCache,
    'coll_intuitive'        : kCollCacheIntuitive,
    'partition'             : kPartitionCache,
    'part_rep'              : kPartRepCache,
    'rep'                   : kRepCache,
    'coll_cache_asymm_link' : kCollCacheAsymmLink,
    'clique_part'           : kCliquePart,
    'clique_part_by_degree' : kCliquePartByDegree,
}
    

def generate_config(run_config):
    config = {}
    config["cache_percentage"] = run_config['cache_percentage']
    config["_cache_policy"] = cache_policy_map[run_config['cache_policy']]
    config["num_device"] = run_config['num_worker']
    config["num_global_step_per_epoch"] = run_config['num_worker'] * run_config['local_step']
    config["num_epoch"] = run_config['epochs']
    config["omp_thread_num"] = run_config['omp_thread_num']
    return config