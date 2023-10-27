import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.layers import GRU,  LSTM, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Reshape
import utils

#get rate from review summary and title info
def LSTUR(filtered_df):
    df_LSTUR = filtered_df
    reviews, meta = utils.getData()
    filtered_df_LSTUR = reviews[reviews[['asin', 'reviewerID', 'overall']].apply(tuple, axis=1).isin(df_LSTUR[['asin', 'reviewerID', 'overall']].apply(tuple, axis=1))]
    df_LSTUR_user = filtered_df_LSTUR[['asin', 'reviewerID', 'overall', 'summary']]
    
    filtered_df_LSTUR_item = meta[meta['asin'].isin(df_LSTUR['asin'])]
    df_LSTUR_item = meta[['asin', 'title']]

    # df_LSTUR_item['title_length'] = df_LSTUR_item['title'].apply(lambda x: len(str(x)))
    # max_title_length = df_LSTUR_item['title_length'].max()
    # avg_title_length = df_LSTUR_item['title_length'].mean()

    # # 计算 summary 列的最大和平均长度
    # df_LSTUR_user['summary_length'] = df_LSTUR_user['summary'].apply(lambda x: len(str(x)))
    # max_summary_length = df_LSTUR_user['summary_length'].max()
    # avg_summary_length = df_LSTUR_user['summary_length'].mean()

    df1 = df_LSTUR_user
    df2 = df_LSTUR_item
    # 创建用户和商品的ID映射
    user_mapping = {user_id: i for i, user_id in enumerate(df1['reviewerID'].unique())}
    item_mapping = {item_id: i for i, item_id in enumerate(df1['asin'].unique())}

    # 添加用户和商品的ID列
    df1['user_id'] = df1['reviewerID'].map(user_mapping)
    df1['item_id'] = df1['asin'].map(item_mapping)


    # 商品文本信息处理
    max_sequence_length = 8  # 根据实际情况调整
    tokenizer_1 = Tokenizer()
    tokenizer_1.fit_on_texts(df2['title'])
    train_text_sequences = pad_sequences(tokenizer_1.texts_to_sequences(df2['title']), maxlen=max_sequence_length)

    merged = pd.merge(df1, df2, on = 'asin')
    # 划分训练集和测试集
    train, test = train_test_split(merged, test_size=0.2, random_state=42)

   
    model = create_model(len(user_mapping), len(item_mapping), max_sequence_length)
    model.fit(
        [train['user_id'], train['item_id'],  train['title']], 
        train['overall'],  
        epochs=500,
        batch_size=32)
    
    test_user_ids = test['user_id'].unique()
    mae_scores = []
    recommend = {}

    for user_id in test_user_ids:
        top_k_items = recommend_top_k(model, user_id,item_mapping,filtered_df_LSTUR )
        recommend[user_id]= top_k_items

        # true_ratings = test[(test['user_id'] == user_id) & (test['item_id'].isin(top_k_items))]['label']
        # predicted_ratings = model.predict([np.array([user_id] * len(true_ratings)), np.array(top_k_items), np.array([get_text_sequence_for_item(item_id) for item_id in top_k_items])]).flatten()
        # mae_scores.append(mean_absolute_error(true_ratings, predicted_ratings))

    # average_mae = np.mean(mae_scores)
    # print(f'Average MAE: {average_mae}')
    result = utils.map_recommendations_back(recommend, user_mapping, item_mapping)
    return result



def create_model(user_count, item_count, embedding_dim=50, gru_units=50, max_sequence_length=100):
    user_input = Input(shape=(1,))
    item_input = Input(shape=(1,))
    item_text_input = Input(shape=(max_sequence_length,))

    user_embedding = Embedding(user_count, embedding_dim)(user_input)
    item_embedding = Embedding(item_count, embedding_dim)(item_input)
    reshaped_item_emb = Reshape((embedding_dim, 1))(item_embedding)  

    user_gru = GRU(gru_units)(user_embedding)
    
    item_gru = Bidirectional(GRU(gru_units))(reshaped_item_emb)  

    # 融合长短期用户表示
    user_representation = user_gru

    # 模型结合
    concat = concatenate([user_representation, item_gru])

    output = Dense(1, activation='sigmoid')(concat)

    model = Model(inputs=[user_input, item_input, item_text_input], outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    return model

def recommend_top_k(model, user_id, k=50, item_mapping= item_mapping, merged = filtered_df_LSTUR):
    all_items = np.array(list(item_mapping.values()))
    user_ids_input = np.array([user_id] * len(all_items))
    item_ids_input = np.array(all_items)
    
    user_text_input = np.array([
        merged[merged['asin'] == item_id]['title'].iloc[0] if len(merged[merged['asin'] == item_id]) > 0 else ''
        for item_id in all_items
    ])
    # 预测评分
    scores = model.predict([user_ids_input, item_ids_input, user_text_input]).flatten()

    # 获取前k个评分最大的商品
    top_k_items = [all_items[i] for i in np.argsort(scores)[-k:]]
    return top_k_items


if __name__ == '__main__':
    db_path  = 'dataset/recommendDB.db'
    engine = create_engine(f'sqlite:///{db_path}')

    reviews, meta = utils.getData()
    merged  = utils.merge(reviews, meta)
    filtered_df = utils.filter_sample(merged, frac=0.2, extract = 'all', sample = 'even')
    filtered_df = utils.rate_std(filtered_df)

    LSTUR = LSTUR(filtered_df, num = 50)

    LSTUR.to_sql('LSTUR', engine, index = False, if_exists = 'replace')
   
    
