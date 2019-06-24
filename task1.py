import collections
import itertools
import os
import sys
import time
from random import randint

from pyspark import SparkContext, SparkConf


def get_jaccard_similarity(x, y):
    s1 = set(x)
    s2 = set(y)
    return len(s1 & s2) / float(len(s1 | s2))


def prepare_data(input_file_name):
    input_file = sc.textFile(input_file_name)
    rdd_by_user = input_file.map(lambda row: row.split(',')) \
        .filter(lambda row: 'user_id' not in row) \
        .map(lambda row: (row[0], [row[1]])) \
        .reduceByKey(lambda x, y: x + y)

    rdd_by_business = input_file.map(lambda x: x.split(',')) \
        .filter(lambda x: 'user_id' not in x) \
        .map(lambda x: (x[1], [x[0]])) \
        .reduceByKey(lambda x, y: x + y)

    list_by_user = rdd_by_user.sortByKey(ascending=True).collect()
    list_by_business = rdd_by_business.sortByKey(ascending=True).collect()

    return list_by_user, list_by_business


def make_min_hash(num_min_hash, user_list, business_list):
    min_hash = []
    users_count = len(user_list)
    businesses_count = len(business_list)
    m = users_count
    for index in range(0, num_min_hash):
        # (ax+b)%m
        a = randint(0, 1000)
        b = randint(0, 1000)

        hash_values = {}
        for mid in range(businesses_count):
            hash_values[business_list[mid][0]] = m

        for user_index in range(users_count):
            user_min_hash = (a * user_index + b) % m
            for business_sub_index in user_list[user_index][1]:
                if hash_values[business_sub_index] > user_min_hash:
                    hash_values[business_sub_index] = user_min_hash

        min_hash.append(hash_values)

    return min_hash


def make_band_list(rdd_min_hash, num_min_hash, bands_num):
    bands_row = int(num_min_hash / bands_num)

    result = rdd_min_hash.map(lambda x: [(k, [x[k]]) for k in x]).reduce(lambda x, y: x + y)

    rdd_band = sc.parallelize(result)\
        .reduceByKey(lambda x, y: x + y)\
        .map(lambda x: [[(x[0], x[1][i]) for i in range(itr, itr + 3)]
                        for itr in range(0, num_min_hash - bands_row + 1, bands_row)])

    return rdd_band.collect()


conf = SparkConf().setAppName("Task1").setMaster("local[*]")
sc = SparkContext(conf=conf)

start_time = time.time()

#
# Prepare Data
#

user_list, business_list = prepare_data(sys.argv[1])
business_list_dict = dict(business_list)

#
# Make hash table
#

num_min_hash = 60
bands_num = 20
min_hash = make_min_hash(num_min_hash, user_list, business_list)
rdd_min_hash = sc.parallelize(min_hash)

#
# Make band list
#

band_list = make_band_list(rdd_min_hash, num_min_hash, bands_num)

#
# Make bucket list
#

bucket_list = {}

for index in range(0, bands_num):
    buckets = {}
    for bands in band_list:
        mid = bands[index][0][0]
        str_to_hash = ''
        for each in bands[index]:
            str_to_hash += ''.join(str(each[1]))

        hash_bin = hash(str_to_hash)

        if hash_bin not in buckets:
            buckets[hash_bin] = [mid]
        else:
            buckets[hash_bin] += [mid]

    bucket_list[index] = list(buckets.values())
    del buckets

rdd_bucket_list = sc.parallelize(list(bucket_list.values())).reduce(lambda x, y: x + y)

#
# Make pairs
#

final_pairs = sc.parallelize(rdd_bucket_list).filter(lambda x: len(x) > 1).map(lambda x: list(set(x))).map(
    lambda x: list(itertools.combinations(x, 2))).reduce(lambda x, y: x + y)

#
# Filter by similarity
#

similarity = {}
for pair in final_pairs:
    a = pair[0]
    b = pair[1]

    l1 = business_list_dict[pair[0]]
    l2 = business_list_dict[pair[1]]
    jaccard_value = get_jaccard_similarity(l1, l2)

    if jaccard_value >= 0.5:
        if a < b:
            if (a, b) not in similarity:
                similarity[(a, b)] = jaccard_value
        else:
            if (b, a) not in similarity:
                similarity[(b, a)] = jaccard_value


#
# Print time taken
#

print("Time Taken")
print(time.time() - start_time)

#
# Write the result to output file
#

ordered_similarity = collections.OrderedDict(sorted(similarity.items()))
output_file = open(sys.argv[2], "w+")
output_file.write("business_id_1, business_id_2, similarity")
output_file.write("\n")
for key, value in ordered_similarity.items():
    output_file.write(str(key[0]) + "," + str(key[1]) + "," + str(value))
    output_file.write("\n")

output_file.close()
