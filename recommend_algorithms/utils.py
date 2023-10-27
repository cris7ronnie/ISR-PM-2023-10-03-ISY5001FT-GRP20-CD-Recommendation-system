import os
import pandas as pd
import gzip
import json


def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

def getData():
  reviews = getDF('dataset/CDs_and_Vinyl.json.gz')
  meta =  getDF('dataset/mata_CDs_and_Vinyl.json.gz')
#  merged = pd.merge(reviews, meta, on = 'asin', how = 'inner')
  return reviews, meta

def merge(reviews, meta):
    meta_cate = meta.explode('category')
    meta_cate.reset_index(drop=True, inplace=True)
    meta_cate_filter = meta_cate[~meta_cate['category'].apply(lambda x: 'CDs & Vinyl' in x)]

    merged = pd.merge(reviews, meta_cate_filter, on = 'asin', how = 'inner')
    selected_columns = ['asin', 'brand', 'category' ]
    merged = merged[selected_columns]
    return merged