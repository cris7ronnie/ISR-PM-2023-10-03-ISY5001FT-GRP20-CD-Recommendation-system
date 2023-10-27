import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
import utils
 
# nn get implicited rating
def NCF(filtered_df, num = 100):
    df = filtered_df
    user_ids = df['reviewerID'].unique()
    item_ids = df['asin'].unique()

    user_id_map = {id: index for index, id in enumerate(user_ids)}
    item_id_map = {id: index for index, id in enumerate(item_ids)}

    num_users = len(user_ids)
    num_items = len(item_ids)

    df['user'] = df['reviewerID'].map(user_id_map)
    df['item'] = df['asin'].map(item_id_map)

    train, test = train_test_split(df, test_size=0.2, random_state=42)

    embedding_dim = 50 

    user_input = Input(shape=(1,), name='user_input')
    item_input = Input(shape=(1,), name='item_input')

    user_embedding = Embedding(input_dim=num_users, output_dim=embedding_dim)(user_input)
    item_embedding = Embedding(input_dim=num_items, output_dim=embedding_dim)(item_input)

    user_flatten = Flatten()(user_embedding)
    item_flatten = Flatten()(item_embedding)

    concat = Concatenate()([user_flatten, item_flatten])
    dense1 = Dense(64, activation='relu')(concat)
    dense2 = Dense(32, activation='relu')(dense1)
    output = Dense(1, activation='sigmoid')(dense2)

    model_NCF = Model(inputs=[user_input, item_input], outputs=output)


    model_NCF.compile(optimizer=Adam(learning_rate=0.001),  loss='mae', metrics=['mae'])

    # 模型训练
    model_NCF.fit([train['user'], train['item']], train['overall'], epochs=10, batch_size=64, validation_split=0.2)

    # 模型评估
    test_loss, test_acc = model_NCF.evaluate([test['user'], test['item']], test['overall'])
    print(f'Test Accuracy: {test_acc}')

    # 进行预测
    predictions = model_NCF.predict([test['user'], test['item']])

    # 获取每个用户的前50个评分最大的商品
    test_user_ids = np.unique(test['user'])
    user_recommendation = []

    for user_id in test_user_ids:
        all_items = np.array(list(set(test['item'])))
        user_ids_input = np.array([user_id] * len(all_items))
        item_ids_input = np.array(all_items)

        # 预测评分
        user_preds = model_NCF.predict([user_ids_input, item_ids_input]).flatten()
        # predictions = model.predict([test['user'], test['item']])

        # 获取前50个评分最大的商品
        top_50_items = [all_items[i] for i in np.argsort(user_preds)[-num:]]
        # predictions.append(top_50_items)
        
        # user_recommendation[user_id] = top_50_items
        user_recommendation[user_id] = top_50_items
        # predictions.append(user_recommendation)

    result = utils.map_recommendations_back(user_recommendation, user_id_map, item_id_map)
    
    return result


if __name__ == '__main__':
    db_path  = 'dataset/recommendDB.db'
    engine = create_engine(f'sqlite:///{db_path}')

    reviews, meta = utils.getData()
    merged  = utils.merge(reviews, meta)
    filtered_df = utils.filter_sample(reviews, frac=0.2, extract = 'all', sample = 'even')
    filtered_df = utils.rate_std(filtered_df)

    NLP_CF = NCF(filtered_df, num = 50)

    NLP_CF.to_sql('NLP_CF', engine, index = False, if_exists = 'replace')
   
    