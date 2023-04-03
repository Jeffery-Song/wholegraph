
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
    

def generate_config(options):
    config = {}
    config["cache_percentage"] = options.cache_percentage
    config["_cache_policy"] = cache_policy_map[options.cache_policy]
    config["num_device"] = options.num_workers
    config["num_global_step_per_epoch"] = options.epochs * options.local_step
    config["num_epoch"] = options.epochs
    # config["num_total_item"] = options.num_total_item
    config["omp_thread_num"] = options.omp_thread_num
    return config