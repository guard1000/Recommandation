import pandas as pd
#%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances  #cosine 유사도 구할때 사용

#유사도 기반 예측 함수 - 함수 정의 부분 이해 안되면 아래부터 보기
def predict(ratings, similarity, type='user'):  #user-item 꼴의 matrix, 유사도, 기준(user기준인지 item기준인지 여부)
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1) #행 평균
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])  #np.newaxis는 같은 배열에 대해 1차원만 증가시킴. 2차원으로 맞춰서 계산하려고 씀
        #dot()은 vector와 matrix의 곱을 구하기 위해 사용
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred


# pandas를 이용해 데이터를 읽어오자.
# user 정보, user가 평점 준 정보, 영화의 정보 파일 읽어옴.
#Reading users file:
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols,encoding='latin-1')

#Reading ratings file:
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols,encoding='latin-1')

#Reading items file: 1과 0으로 특성을 지녔는지 여부만 표기. 24개 특성
i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols,encoding='latin-1')

#train용 데이터와 test용 데이터(미리 분류해 둠) 읽어오기
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_train = pd.read_csv('ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')

'''
#각 읽어온 파일별 shape이나 내용 궁금하면 확인
print(ratings_train.shape, ratings_test.shape)

print(users.shape)
users.head()

print(ratings.shape)
ratings.head()

print(items.shape)
items.head()
'''

# 유저 수와 영화 수 구하기 (unique)
n_users = ratings.user_id.unique().shape[0]
n_items = ratings.movie_id.unique().shape[0]

# user-item 행렬 만들어서 부여한 평점 값 넣기
data_matrix = np.zeros((n_users, n_items))  #일단 0으로 채워서!
for line in ratings.itertuples():
    data_matrix[line[1]-1, line[2]-1] = line[3]

#유사도를 구해보자 -> 코사인 유사도 활용 -> sklearn의 pairwise_distance function 사용
#user_similarity : user간 유사도
#item_similarity : item간 유사도
user_similarity = pairwise_distances(data_matrix, metric='cosine')
item_similarity = pairwise_distances(data_matrix.T, metric='cosine')    #data_matrix.T 는 행과 열을 바꿈!

#예측 때리기(맨 위에 선언해 둔 함수 호출)
user_prediction = predict(data_matrix, user_similarity, type='user')
item_prediction = predict(data_matrix, item_similarity, type='item')

