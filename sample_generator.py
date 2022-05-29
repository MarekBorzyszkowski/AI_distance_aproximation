from sklearn.utils import shuffle

import csv
import random
from scipy.spatial import distance
import numpy as np


def save_to_csv(result, file_name, folder_name, numbers_per_point):
    header = ['a{}'.format(i) for i in range(1, numbers_per_point + 1)] + \
             ['b{}'.format(i) for i in range(1, numbers_per_point + 1)] + ['distance']

    with open('csv/' + folder_name + '/' + file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(result)


def generate_sample_file(dimension, num_of_samples):
    return generate_samples(dimension, num_of_samples)


def generate_sample_file_with_zeros(dimension, num_of_zeros, num_of_samples):
    samples1 = generate_list_of_points(dimension - num_of_zeros, num_of_samples)
    for s in samples1:
        s.extend([0] * num_of_zeros)
    samples2 = generate_list_of_points(dimension - num_of_zeros, num_of_samples)
    for s in samples2:
        s.extend([0] * num_of_zeros)
    distance_x1_x2 = create_distance_list(dimension, num_of_samples, samples1, samples2)
    result = [samples1[i] + samples2[i] + [distance_x1_x2[i]] for i in range(num_of_samples)]
    return result


def create_distance_list(dimension, num_of_samples, samples1, samples2):
    distance_x1_x2 = []
    for i in range(num_of_samples):
        distance_x1_x2.append(distance_measure(samples1[i], samples2[i]) * (dimension ** 0.5) / dimension)
        if distance_x1_x2[i] >= 1.0:
            distance_x1_x2[i] = 1.0
    return distance_x1_x2


def distance_measure(list1, list2):
    sum = 0
    for i in range(len(list1)):
        sum += (list1[i] - list2[i]) ** 2
    return sum ** 0.5


def generate_samples_with_norms(dimension, num_of_samples):
    samples = generate_samples(dimension, num_of_samples)
    return norm_samples(dimension, samples)


def generate_samples(dimension, num_of_samples):
    list_of_x1 = generate_list_of_points(dimension, num_of_samples)
    list_of_x2 = generate_list_of_points(dimension, num_of_samples)
    distance_x1_x2 = create_distance_list(dimension, num_of_samples, list_of_x1, list_of_x2)
    result = [list_of_x1[i] + list_of_x2[i] + [distance_x1_x2[i]] for i in range(num_of_samples)]
    validate_data(result)
    return result


def generate_list_of_points(row_length, num_of_samples):
    return [[random.uniform(0, 1.0) for j in range(row_length)] for i in range(num_of_samples)]


def norm_samples(dimension, samples):
    return [[average(points[0:dimension]), euclidean_norm(points[0:dimension]), inf_norm(points[0:dimension]),
             average(points[dimension:2 * dimension]), euclidean_norm(points[dimension:2 * dimension]),
             inf_norm(points[dimension:2 * dimension]), points[-1]
             ] for points in samples]


def average(sample):
    return sum(sample) / len(sample)


def euclidean_norm(sample):
    return np.linalg.norm(sample) / len(sample)


def inf_norm(sample):
    return max(sample)


def abs_a_minus_b_samples():
    result = []
    for a in range(100):
        for b in range(100):
            result.append([a / 100.0, b / 100.0, abs(a - b) / 100.0])
    validate_data(result)
    return result


def abs_a_minus_b_samples_random(num_of_samples):
    result = [[random.uniform(0, 1.0) for j in range(2)] for i in range(num_of_samples)]
    for row in result:
        row.append(abs(row[0] - row[1]))
    validate_data(result)
    return result


def x_squared_samples():
    result = [[i / 100, (i / 100) ** 2] for i in range(100)]
    validate_data(result)
    return result


def x_squared_samples_random(num_of_samples):
    result = [[random.uniform(0, 1.0)] for i in range(num_of_samples)]
    for row in result:
        row.append(row[0] ** 2)
    validate_data(result)
    return result


def generate_discrete_samples_3d():
    points1 = []
    points2 = []
    for a in range(10):
        for b in range(10):
            for c in range(10):
                for d in range(10):
                    for e in range(10):
                        for f in range(10):
                            points1.append([a / 10.0, b / 10.0, c/10.0])
                            points2.append([d/10.0, e/10.0, f/10.0])
    distance_value = create_distance_list(3, 10 ** 6, points1, points2)
    result = [points1[i] + points2[i] + [distance_value[i]] for i in range(10 ** 6)]
    return result


def validate_data(result):
    for row in result:
        for num in row:
            if num >= 1.0:
                num = 1.0

# generate_sample_file(3, 1000, 'test')

# generate_sample_file_with_zeros(3, 1, 1000, 'test')

# result = []
# for i in range(1, 21):
#     result.extend(generate_samples_with_norms(i, 100000))
#     print(i)
# result = np.array(result)
# result = shuffle(result)
# save_to_csv(result, 'dmn2000000norm.csv', 'training', 3)
# for i in range(5):
#     save_to_csv(generate_samples(3, 1000000), 'd3n1000000_s{}.csv'.format(i), 'training/3DimBatch', 3)

save_to_csv(generate_discrete_samples_3d(),'d3_1000000.csv', 'training/3DimDiscrete', 3)
