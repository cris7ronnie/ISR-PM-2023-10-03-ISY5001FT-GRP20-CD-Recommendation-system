import utils
import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler


def most_rate(review, merged):
    ave_rating = reviews.groupby('asin')['overall'].mean().reset_index(name = 'ave_rating')
    rating = pd.merge(ave_rating, merged, on = 'asin',how = 'inner')
    return rating

def get_rank(meta):
    df['Rank'] = df['SalesRank'].str.extract(r'(\d[\d,]*)').str.replace(',', '').astype(int)

def get_top(data, col_name= ''):
    if not col_name:
        data_counts = data['asin'].value_counts().reset_index(name = col_name+'_counts')
    else:
        data_counts = data.groupby('asin')[col_name].nunique().reset_index(name = col_name+'_counts')
    data_counts_top_100 = data_counts[data_counts[col_name+'_counts']>1000]
    
    scaler = MinMaxScaler()
    data_counts[col_name+'_counts'] = scaler.fit_transform(data_counts[[col_name+'_counts']]) 
#    data_counts_top_5000 = data_counts.sort_values(by=col_name+'_counts', ascending=False).head(5000)
    
    return data_counts


def most_sale(reviews, merged):
    review_top = get_top(reviews)
    reviewer_top = get_top(reviews,'reviewerID' )
    top_merged = pd.merge(review_top, reviewer_top, on = 'asin',how = 'inner')
    top_merged['total'] = 0.5*top_merged['_counts']+ 0.5*top_merged['reviewerID_counts']
# rank还没写
#   top_result = utils.merge(top_merged, meta)
    top_result = pd.merge(top_result, merged, on = 'asin',how = 'inner')
    return top_result



if __name__ == '__main__':
    db_path  = 'dataset/recommendDB.db'
    engine = create_engine(f'sqlite:///{db_path}')

    reviews, meta = utils.getData()
    merged  = utils.merge(reviews, meta)

    most_rating = most_rate(reviews, merged)
    most_rating.to_sql('most_rating', engine, index = False, if_exists = 'replace')
   
    most_saling = most_sale(reviews, meta)
    most_saling.to_sql('most_saling', engine, index = False, if_exists = 'replace')
