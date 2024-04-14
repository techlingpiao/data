import pandas as pd
import pickle
from scipy.sparse import csr_matrix

anime_data = pd.read_csv('anime.csv')
user_data = pd.read_csv('rating_complete.csv', low_memory=False)

anime_ids = anime_data['MAL_ID'].unique().tolist()
anime2_encoded = {x: i for i, x in enumerate(anime_ids)}
encoded2_anime = {i: x for i, x in enumerate(anime_ids)}
anime_data["anime"] = anime_data["MAL_ID"].map(anime2_encoded)

user_ids = user_data["user_id"].unique().tolist()
user2_encoded = {x: i for i, x in enumerate(user_ids)}
encoded2_user = {i: x for i, x in enumerate(user_ids)}
user_data["user"] = user_data["user_id"].map(user2_encoded)

user_data = user_data.merge(anime_data, left_on="anime_id", right_on="MAL_ID", how='inner')
user_data = user_data[['user_id', 'anime_id', 'rating', 'user', 'anime']]

user_data = user_data.sort_values(['user', 'anime'], ascending=[True, True])

user_data_2 = user_data[user_data['rating'] >= 7]
user_data_2 = user_data_2[['user', 'anime']]
user_data_2.insert(user_data_2.shape[1], 'click', 1)

m = len(user_data_2) // 30
user_data = user_data_2[0:m]

anime_ids = user_data["anime"].unique().tolist()
user_ids = user_data["user"].unique().tolist()

density = len(user_data) / (len(anime_ids) * len(user_ids))

train = pd.DataFrame(columns=['user', 'anime', 'click'])
test = pd.DataFrame(columns=['user', 'anime', 'click'])
val = pd.DataFrame(columns=['user', 'anime', 'click'])

for id in user_ids:
    df = user_data[user_data['user'] == id]
    train_len = int(len(df) * 0.7)
    test_len = int(len(df) * 0.2)
    new_train = df.iloc[:train_len]
    new_test = df.iloc[train_len:train_len + test_len]
    new_val = df.iloc[train_len + test_len:]
    train = train.append(new_train, ignore_index=True)
    test = test.append(new_test, ignore_index=True)
    val = val.append(new_val, ignore_index=True)

user_item_matrix_train = csr_matrix((train['click'], (train['user'], train['anime'])), shape=(len(user_ids), len(anime_ids)))
user_item_matrix_test = csr_matrix((test['click'], (test['user'], test['anime'])), shape=(len(user_ids), len(anime_ids)))
user_item_matrix_val = csr_matrix((val['click'], (val['user'], val['anime'])), shape=(len(user_ids), len(anime_ids)))

with open('train_mat.pkl', 'wb') as f:
    pickle.dump(user_item_matrix_train, f)

with open('valid_mat.pkl', 'wb') as f:
    pickle.dump(user_item_matrix_val, f)

with open('test_mat.pkl', 'wb') as f:
    pickle.dump(user_item_matrix_test, f)
