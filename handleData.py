import pandas as pd
import pickle
from scipy.sparse import csr_matrix


anime_data = pd.read_csv('anime.csv')
user_data = pd.read_csv('rating_complete.csv', low_memory=False)

# rearrange anime_data via unique anime_id
anime_ids = anime_data['MAL_ID'].unique().tolist()
anime2_encoded = {x: i for i, x in enumerate(anime_ids)}
encoded2_anime = {i: x for i, x in enumerate(anime_ids)}
anime_data["anime"] = anime_data["MAL_ID"].map(anime2_encoded)

# rearrange user_data via unique user_id
user_ids = user_data["user_id"].unique().tolist()
user2_encoded = {x: i for i, x in enumerate(user_ids)}
encoded2_user = {i: x for i, x in enumerate(user_ids)}
user_data["user"] = user_data["user_id"].map(user2_encoded)

# merge two table sets
user_data = user_data.merge(anime_data,left_on="anime_id",right_on="MAL_ID",how='inner')
user_data=user_data[['user_id','anime_id','rating','user','anime']]

# sort by user, anime
user_data = user_data.sort_values(['user','anime'], ascending = [True,True])

# original user data3
# user_data_1 = user_data[['user','anime']]
# user_data_1.insert(user_data_1.shape[1], 'click', 1)

# user data with high ratings (>7)
user_data_2 = user_data[(user_data['rating']>=7)]
user_data_2 = user_data_2[['user','anime']]
# user_data_2.insert(user_data_2.shape[1], 'click', 1)

m = len(user_data_2)//30
user_data = user_data_2[0:m]

anime_ids = user_data["anime"].unique().tolist()
user_ids = user_data["user"].unique().tolist()
# print("The number of users:", len(user_ids))
# print("The number of animes:", len(anime_ids))
# print("The number of rated blocks:", len(user_data))

density2 = len(user_data)/(len(anime_ids)*len(user_ids))
# print("Density of final data:", density2)

# train = pd.DataFrame(columns=['user', 'anime', 'click'])
# test = pd.DataFrame(columns=['user', 'anime', 'click'])
# val = pd.DataFrame(columns=['user', 'anime', 'click'])
#
# for id in user_ids:
#     df = user_data[(user_data['user']==id)]
#     train_len = int(len(df)*0.7)
#     test_len = int(len(df)*0.2)
#     new_train = df.iloc[0:train_len]
#     new_test = df.iloc[train_len: train_len+test_len]
#     new_val = df.iloc[train_len+test_len:]
#     train = train.append(new_train, ignore_index = True)
#     test = test.append(new_test, ignore_index = True)
#     val = test.append(new_val, ignore_index=True)


# 创建用户-项目的交互矩阵
row = user_data['user'].map(user2_encoded)
col = user_data['anime'].map(anime2_encoded)
data = [1] * len(user_data)
user_item_matrix = csr_matrix((data, (row, col)), shape=(len(user_ids), len(anime_ids)))

# 划分数据集
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

total_size = len(user_data)
train_size = int(total_size * train_ratio)
val_size = int(total_size * val_ratio)
test_size = total_size - train_size - val_size

train = user_item_matrix[:train_size]
val = user_item_matrix[train_size:train_size+val_size]
test = user_item_matrix[train_size+val_size:total_size]

# 分别保存训练集、验证集和测试集
with open('train_mat.pkl', 'wb') as f:
    pickle.dump(train.toarray(), f)

with open('valid_mat.pkl', 'wb') as f:
    pickle.dump(val.toarray(), f)

with open('test_mat.pkl', 'wb') as f:
    pickle.dump(test.toarray(), f)
