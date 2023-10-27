#settings
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import Dataset, Reader, SVD
from surprise import accuracy
from surprise import Reader
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate
import numpy as np
import pandas as pd

import utils

def cf_SVD(filtered_df, num = 100):
    df = filtered_df
    user_ids = df['reviewerID'].unique()
    item_ids = df['asin'].unique()

    user_id_map = {id: index for index, id in enumerate(user_ids)}
    item_id_map = {id: index for index, id in enumerate(item_ids)}
    
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(filtered_df, reader)
    trainset, testset = train_test_split(data, test_size=0.25, random_state=42)

    model_SVD = SVD()
    model_SVD.fit(trainset)

    cross_validate_results = cross_validate(model_SVD, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    print(f"Mean RMSE: {cross_validate_results['test_rmse'].mean()}")
    print(f"Mean MAE: {cross_validate_results['test_mae'].mean()}")
   
    user_ids = trainset.all_users()
    top500_recommendations = {}

    for user_id in user_ids:
        # 获取用户未评分的物品
        user_items = [item for item in trainset.all_items() if user_id not in trainset.ur[item]]

        # 预测评分
        predictions = [model_SVD.predict(user_id, item) for item in user_items]

        # 按预测评分从高到低排序
        sorted_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)

        # 获取前500个推荐的物品
        top500_recommendations[user_id] = [pred.iid for pred in sorted_predictions[: num]]


    # 输出每个用户的前500个推荐
    for user_id, recommendations in top500_recommendations.items():
        print(f"User {user_id}: {recommendations}")
    
    # user_item_tuples = [(user_id, item) for user_id, recommended_items in top500_recommendations.items() for item in recommended_items]
    # result = pd.DataFrame(user_item_tuples, columns=['user_id', 'recommended_item'])
    result = utils.map_recommendations_back(top500_recommendations, user_id_map, item_id_map)
    return result


def cf_KNN(filtered_df, num = 100):
    df = filtered_df
    user_ids = df['reviewerID'].unique()
    item_ids = df['asin'].unique()

    user_id_map = {id: index for index, id in enumerate(user_ids)}
    item_id_map = {id: index for index, id in enumerate(item_ids)}
    
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(filtered_df, reader)
    trainset, testset = train_test_split(data, test_size=0.25, random_state=42)
  
    algo = KNNBasic(k=40,sim_options={'name': 'pearson'})   
    algo.fit(trainset)

    cross_validate_results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    print(f"Mean RMSE: {cross_validate_results['test_rmse'].mean()}")
    print(f"Mean MAE: {cross_validate_results['test_mae'].mean()}")

    user_ids = trainset.all_users()
    top500_recommendations = {}
    for user_id in user_ids:
        # 获取用户未评分的物品
        user_items = [item for item in trainset.all_items() if user_id not in trainset.ur[item]]

        # 预测评分
        predictions = [algo.predict(user_id, item) for item in user_items]

        # 按预测评分从高到低排序
        sorted_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)

        # 获取前500个推荐的物品
        top500_recommendations[user_id] = [pred.iid for pred in sorted_predictions[:num]]


    # 输出每个用户的前500个推荐
    for user_id, recommendations in top500_recommendations.items():
        print(f"User {user_id}: {recommendations}")
       
    # user_item_tuples = [(user_id, item) for user_id, recommended_items in top500_recommendations.items() for item in recommended_items]
    # result = pd.DataFrame(user_item_tuples, columns=['user_id', 'recommended_item'])
    result = utils.map_recommendations_back(top500_recommendations, user_id_map, item_id_map)
    return result

    # user_item_tuples = [(user_id, item) for user_id, recommended_items in top500_recommendations.items() for item in recommended_items]
    # result = pd.DataFrame(user_item_tuples, columns=['user_id', 'recommended_item'])
    # return result





if __name__ == '__main__':
    reviews, meta = utils.getData()
    filtered_df = utils.filter_sample(reviews, frac = 0.1)
    filtered_df = utils.rate_std(filtered_df)
    
    db_path  = 'dataset/recommendDB.db'
    engine = create_engine(f'sqlite:///{db_path}')

    
    merged  = utils.merge(reviews, meta)

    most_rating = most_rate(reviews, merged)
    most_rating.to_sql('most_rating', engine, index = False, if_exists = 'replace') 
