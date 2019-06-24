import math
import os
import sys
import time
from itertools import combinations

from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS

train_file_name = sys.argv[1]
test_file_name = sys.argv[2]
case_id = int(sys.argv[3])
output_file_name = sys.argv[4]

start_time = time.time()

conf = SparkConf().setAppName("Task2").setMaster("local[*]")
sc = SparkContext(conf=conf)

num_partitions = 2


def quit_script(err_desc):
    raise SystemExit('\n' + "PROGRAM EXIT: " + err_desc + ', please check your input' + '\n')


# Check if file exist
if not os.path.isfile(train_file_name):
    quit_script("Train file doesn't exist or it's not a file")

if not os.path.isfile(test_file_name):
    quit_script("Test file doesn't exist or it's not a file")

if 1 > case_id > 3:
    quit_script("Case id should be from 1 to 3")


def get_raw_ratings_rdd(raw_ratings):
    return raw_ratings.map(lambda row: row.split(',')) \
        .filter(lambda row: 'user_id' not in row) \
        .map(lambda row: (row[0], row[1], float(row[2])))


train_ratings = sc.textFile(train_file_name).repartition(num_partitions)
test_ratings = sc.textFile(test_file_name).repartition(num_partitions)


def calculate_rmse(predicted_rdd, actual_rdd):
    predicted_reformatted_rdd = predicted_rdd.map(lambda row: ((row[0], row[1]), row[2]))
    actual_reformatted_rdd = actual_rdd.map(lambda row: ((row[0], row[1]), row[2]))

    squared_errors_rdd = (predicted_reformatted_rdd.join(actual_reformatted_rdd).map(
        lambda row: (row[1][1] - row[1][0]) ** 2))

    if squared_errors_rdd.isEmpty():
        total_error = 0
    else:
        total_error = squared_errors_rdd.reduce(lambda x, y: x + y)

    num_ratings = squared_errors_rdd.count()

    if num_ratings == 0:
        return 0

    return math.sqrt(float(total_error) / num_ratings)


output_file = open(output_file_name, "w+")
if case_id == 2:
    def finding_user_pairs(users_with_rating):
        for user1, user2 in combinations(users_with_rating, 2):
            return (user1[0], user2[0]), (user1[1], user2[1])


    def cosine_sim(user_pair, rating_pairs):
        sum_x, sum_xy, sum_y, x = (0.0, 0.0, 0.0, 0)

        for rating_pair in rating_pairs:
            sum_x += float(rating_pair[0]) * float(rating_pair[0])
            sum_y += float(rating_pair[1]) * float(rating_pair[1])
            sum_xy += float(rating_pair[0]) * float(rating_pair[1])

            x += 1

        cos_sim = cosine(sum_xy, math.sqrt(sum_x), math.sqrt(sum_y))
        return user_pair, (cos_sim, x)


    def cosine(dot_product, rating1_norm_squared, rating2_norm_squared):
        num = dot_product
        den = rating1_norm_squared * rating2_norm_squared

        return (num / (float(den))) if den else 0.0


    def key_of_first_user(user_pair, business_sim_data):
        (user1_id, user2_id) = user_pair
        return user1_id, (user2_id, business_sim_data)


    def near_neighbour(user, users_and_sims, n):
        users_and_sims.sort(key=lambda x: x[1][0], reverse=True)
        return user, users_and_sims[:n]


    def predict_business_recommendation_rate(user_id, business_id, users_with_rating, user_sims_dict):
        t = 0
        sim_s = 0

        user_sims_for_user = user_sims_dict.get(user_id, None)
        if user_sims_for_user:
            # ([(user_id_2, (cos_sum1, counts1)), (user_id_3, (cos_sum2, counts2))])
            user_sims = dict(user_sims_dict.get(user_id, None))
            for (neigh, user_sims_value) in user_sims.items():

                # lookup the business predictions for this similar neighbours
                unscored_businesss = users_with_rating.get(neigh, None)
                if unscored_businesss:
                    unscored_business = dict(unscored_businesss).get(business_id, None)
                    if user_id != neigh and unscored_business:
                        # update totals and sim_s with the rating data
                        t += user_sims_value[0] * unscored_business
                        sim_s += user_sims_value[0]

        if sim_s == 0:
            score = 0
        else:
            score = t / sim_s

        return user_id, business_id, score
    
    
    business_user_pairs = train_ratings.map(lambda x: x.split(',')) \
        .filter(lambda row: 'user_id' not in row) \
        .map(lambda row: (row[1], [(row[0], float(row[2]))])) \
        .reduceByKey(lambda x, y: x + y)
    paired_users = business_user_pairs.filter(lambda p: len(p[1]) > 1)
    paired_users = paired_users.map(
        lambda p: finding_user_pairs(p[1])).groupByKey()

    user_sim = paired_users.map(
        lambda p: cosine_sim(p[0], p[1]))

    user_sim = user_sim.map(
        lambda p: key_of_first_user(p[0], p[1])).groupByKey()

    k = 10
    user_sim = user_sim.map(lambda x: (x[0], list(x[1]))).map(
        lambda p: near_neighbour(p[0], p[1], k))
    user_sim_dict = dict(user_sim.collect())
    us = sc.broadcast(user_sim_dict)

    user_business_history = train_ratings.map(lambda x: x.split(',')) \
        .filter(lambda row: 'user_id' not in row) \
        .map(lambda row: (row[0], [(row[1], float(row[2]))])) \
        .reduceByKey(lambda x, y: x + y)

    user_dict = dict(user_business_history.collect())
    u = sc.broadcast(user_dict)

    test_line = sc.textFile(test_file_name).repartition(2)
    test_ratings_rdd = get_raw_ratings_rdd(test_line)
    validation_for_predict_rdd = test_ratings_rdd.map(lambda row: (row[0], row[1]))

    user_business_recs = validation_for_predict_rdd.map(
        lambda p: predict_business_recommendation_rate(p[0], p[1], u.value, us.value)).collect()

    output_file.write("user_id, business_id, prediction")
    output_file.write("\n")
    for rating_items in user_business_recs:
        output_file.write(str(rating_items[0]) + "," + str(rating_items[1]) + "," + str(rating_items[2]))
        output_file.write("\n")

    print("Time Taken")
    print(time.time() - start_time)
elif case_id == 3:
    def find_business_pairs(business_with_rating):
        for business1, business2 in combinations(business_with_rating, 2):
            return (business1[0], business2[0]), (business1[1], business2[1])


    def fetch_data(line):
        line = line.split(",")
        return line[0], (line[1], float(line[2]))


    def key_of_first_item(business_pair, business_sim_data):
        (business1_id, business2_id) = business_pair
        return business1_id, (business2_id, business_sim_data)


    def near_neighbors(business_id, business_and_sims, n):
        business_and_sims.sort(key=lambda x: x[1][0], reverse=True)
        return business_id, business_and_sims[:n]


    def cosine_sim(movie_pair, rating_pairs):
        sum_x, sum_xy, sum_y, x = (0.0, 0.0, 0.0, 0)

        for rating_pair in rating_pairs:
            sum_x += float(rating_pair[0]) * float(rating_pair[0])
            sum_y += float(rating_pair[1]) * float(rating_pair[1])
            sum_xy += float(rating_pair[0]) * float(rating_pair[1])

            x += 1

        cosine_sim = cosine(sum_xy, math.sqrt(sum_x), math.sqrt(sum_y))
        return movie_pair, (cosine_sim, x)


    def cosine(dot_product, rating1_norm_squared, rating2_norm_squared):
        num = dot_product
        den = rating1_norm_squared * rating2_norm_squared

        return (num / (float(den))) if den else 0.0


    def predict_all_business_for_user(user_id, business_id, businesss_with_rating, businesses_sims_dict):
        t = 0
        sim_sum = 0

        business_with_rating = businesss_with_rating.get(user_id, None)
        if business_with_rating:
            for (business, rating) in business_with_rating:

                # lookup the nearest neighbors for this business
                near_neigh = businesses_sims_dict.get(business, None)

                if near_neigh:
                    for (neigh, (sim, count)) in near_neigh:
                        if neigh != business and neigh == business_id:
                            t += sim * rating
                            sim_sum += sim

        if sim_sum == 0:
            score = 0
        else:
            score = t / sim_sum

        return user_id, business_id, score
    
    
    user_business_pairs = train_ratings.filter(lambda row: 'user_id' not in row).map(fetch_data).groupByKey().cache()

    # Make pairs for 2 businesses and ratings
    # (business_1, business_2), (rating_1, rating_2)
    # Group by (business_1, business_2)
    paired_businesss = user_business_pairs.filter(
        lambda p: len(p[1]) > 1).map(
        lambda p: find_business_pairs(p[1])).groupByKey()

    # cos_sim = Sigma(rating_1 * rating_2)/(abs(Sigma(rating_1)) * abs(Sigma(rating_2))
    # (business_1, business_2), (cos_sum, counts)
    business_sim = paired_businesss.map(
        lambda p: cosine_sim(p[0], p[1]))

    # (business_1, (business_2, (cos_sum, counts))
    # Group by business_1
    business_sim = business_sim.map(
        lambda p: key_of_first_item(p[0], p[1])).groupByKey()

    # (business_1, [(business_2, (cos_sum1, counts1)), (business_3, (cos_sum2, counts2))])
    # order by cos_sum and take k
    k = 10
    business_sim = business_sim.map(lambda x: (x[0], list(x[1]))).map(
        lambda p: near_neighbors(p[0], p[1], k)).collect()
    business_sim_dict = dict(business_sim)
    bs = sc.broadcast(business_sim_dict)

    # [(user_id, [(business_id, rating), ....]), ...]
    user_business_history = train_ratings.map(lambda x: x.split(',')) \
        .filter(lambda row: 'user_id' not in row) \
        .map(lambda row: (row[0], [(row[1], float(row[2]))])) \
        .reduceByKey(lambda x, y: x + y)

    # {user_id:[(business_id, rating), ...], ...]
    user_dict = dict(user_business_history.collect())
    u = sc.broadcast(user_dict)

    test_line = sc.textFile(test_file_name).repartition(2)
    test_ratings_rdd = get_raw_ratings_rdd(test_line)
    validation_for_predict_rdd = test_ratings_rdd.map(lambda row: (row[0], row[1]))

    user_business_recs = validation_for_predict_rdd \
        .map(lambda p: predict_all_business_for_user(p[0], p[1], u.value, bs.value)).collect()

    output_file.write("user_id, business_id, prediction")
    output_file.write("\n")
    for rating_items in user_business_recs:
        output_file.write(str(rating_items[0]) + "," + str(rating_items[1]) + "," + str(rating_items[2]))
        output_file.write("\n")

    print("Time Taken")
    print(time.time() - start_time)
elif case_id == 1:
    def append_int_ids(raw_ratings):
        raw_ratings_rdd = get_raw_ratings_rdd(raw_ratings)

        int_user_id_mapping_rdd = raw_ratings_rdd.map(lambda row: row[0]).distinct(num_partitions).zipWithUniqueId()
        int_business_id_mapping_rdd = raw_ratings_rdd.map(lambda row: row[1]).distinct(num_partitions).zipWithUniqueId()

        int_ids_appended_rdd = raw_ratings_rdd.keyBy(lambda row: row[0]) \
            .join(int_user_id_mapping_rdd) \
            .map(lambda row: (row[1][1], row[0], row[1][0][1], row[1][0][2])) \
            .keyBy(lambda row: row[2]) \
            .join(int_business_id_mapping_rdd) \
            .map(lambda row: (row[1][0][0], row[1][0][1], row[1][1], row[1][0][2], row[1][0][3]))

        return int_ids_appended_rdd


    def get_only_int_ids_rdd(int_ids_appended_rdd):
        return int_ids_appended_rdd.map(lambda row: (row[0], row[2], row[4]))


    train_int_ids_appended_rdd = append_int_ids(train_ratings)
    train_only_int_ids_rdd = get_only_int_ids_rdd(train_int_ids_appended_rdd)
    test_int_ids_appended_rdd = append_int_ids(test_ratings)
    test_only_int_ids_rdd = get_only_int_ids_rdd(test_int_ids_appended_rdd)
    validation_for_predict_rdd = test_only_int_ids_rdd.map(lambda row: (row[0], row[1]))

    seed = 5
    # iterations = 20
    # rank = 12
    iterations = 5
    rank = 4
    regularization_parameter = 0.1
    tolerance = 0.03

    minError = float('inf')
    bestRank = -1
    bestIteration = -1

    model = ALS.train(train_only_int_ids_rdd, rank, seed=seed, iterations=iterations,
                      lambda_=regularization_parameter)
    predicted_ratings_rdd = model.predictAll(validation_for_predict_rdd)
    # #Calculate RMSE
    # error = calculate_rmse(predicted_ratings_rdd, test_only_int_ids_rdd)
    # print('The RMSE is %s' % error)

    string_id_predict_ratings_rdd = predicted_ratings_rdd \
        .keyBy(lambda row: (row[0], row[1])) \
        .join(test_int_ids_appended_rdd.keyBy(lambda row: (row[0], row[2]))) \
        .map(lambda row: (row[1][1][1], row[1][1][3], row[1][0][2]))

    output_file.write("business_id_1, business_id_2, prediction")
    output_file.write("\n")
    for rating_items in string_id_predict_ratings_rdd.collect():
        output_file.write(str(rating_items[0]) + "," + str(rating_items[1]) + "," + str(rating_items[2]))
        output_file.write("\n")

    print("Time Taken")
    print(time.time() - start_time)

output_file.close()
