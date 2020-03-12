# blobs
{   'MyExpectMax': {   'elapsed': 0.667,
                       'error': 10,
                       'iters': 5,
                       'n_clusters': 6,
                       'random_state': 372,
                       'score': 0.7502547601018824},
    'MyKMeans': {   'elapsed': 0.412,
                    'error': 10,
                    'iters': 6,
                    'n_clusters': 6,
                    'random_state': 372,
                    'score': 0.7502547601018824}}


# wine
{   'MyExpectMax': {   'elapsed': 2.176,
                       'iters': 8,
                       'n_clusters': 3,
                       'random_state': 372,
                       'score': 0.02883871542935671},
    'MyKMeans': {   'elapsed': 1.02,
                    'iters': 10,
                    'n_clusters': 2,
                    'random_state': 25,
                    'score': 0.6238198074799757}}

# Running expect max alone
{   'MyExpectMax': {   'cluster_std': 100,
                       'elapsed': 58.264,
                       'iters': 100,
                       'n_clusters': 7,
                       'random_state': 464,
                       'score': 0.5278521402021635},

# alone again
{   'MyExpectMax': {   'cluster_std': 1000,
                       'elapsed': 8.249,
                       'iters': 28,
                       'n_clusters': 3,
                       'random_state': 464,
                       'score': 0.5876184085629038},
    'MyKMeans': {'score': -1}}

# a good one
{   'MyExpectMax': {   'cluster_std': 1000,
                       'elapsed': 8.603,
                       'iters': 28,
                       'n_clusters': 3,
                       'random_state': 464,
                       'score': 0.5876184085629038},
    'MyKMeans': {   'cluster_std': 1000,
                    'elapsed': 1.111,
                    'iters': 10,
                    'n_clusters': 2,
                    'random_state': 464,
                    'score': 0.6238198074799757}}
total_elapsed: 765.8304150104523
