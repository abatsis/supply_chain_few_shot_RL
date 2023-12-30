def feature_name(key, i):
    if key == 'periods':
        return f'Periods'
    elif key == 'I0':
        return f'Initial inventory at level {i}'
    elif key == 'p':
        return f'Unit price'
    elif key == 'r':
        f'Replenishment costs at level {i}'
    elif key == 'k':
        return f'Backlog cost at level {i}'
    elif key == 'h':
        return f'Holding cost at level {i}'
    elif key == 'c':
        return f'Production capacity at level {i + 1}'
    elif key == 'L':
        return f'Lead time at level {i}'
    elif key == 'backlog':
        return f'Backlog'
    elif key == 'dist':
        return f'Demand distribution'
    elif key == 'dist_param':
        return f'dist parameter {i}'
    elif key == 'alpha':
        return f'Discount factor'
    elif key == 'seed_int':
        return f'Seed'
    else:
        return 'ERROR'


sortedKeys = [
    'I0',
    'L',
    'alpha',
    'backlog',
    'c',
    'dist',
    'dist_param',
    'h',
    'k',
    'p',
    'periods',
    'r',
    'seed_int',
]


def state_feature(i, m):
    bin_number, index = divmod(i, m - 1)
    if bin_number == 0:
        return f'Inventory at level {index}'

    return f'Order at level {index} at time t - {bin_number}'


def get_features(input_sizes,  max_number_of_levels):
    features = []
    for i in range(input_sizes[0]):
        features.append(state_feature(i, max_number_of_levels))

    for key, size in zip(sortedKeys, input_sizes[1:]):
        for i in range(size):
            features.append(feature_name(key, i))

    return features
