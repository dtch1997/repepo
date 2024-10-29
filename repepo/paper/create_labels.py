""" Script to create dataset of labels """

import pandas as pd 

df = pd.read_parquet("llama7b_steerability.parquet.gzip")
df.head()