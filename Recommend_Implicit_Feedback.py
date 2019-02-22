import pandas as pd
import scipy.sparse as sparse
import numpy as np
import random
from scipy.sparse.linalg import spsolve
#import implicit #step4까지 할꺼면
from sklearn import metrics #step5

def make_train(ratings, pct_test=0.2): # 트레이닝셋과 테스트셋 만들어주는 함수 - step2에서 사용
    # ratings: 원래의 daata. 앞서만든 4338x3664 sparse matrix 받아올꺼임
    # pct_test: 원본 데이터에다 Mask(숨김) 처리 할 비율.

    test_set = ratings.copy()  # 원본 copy
    test_set[test_set != 0] = 1  # 테스트셋을 이진 matix로
    training_set = ratings.copy()  # training 세트도 원본 copy
    nonzero_inds = training_set.nonzero()  # interaction exists 존재하는 인덱스 찾음
    nonzero_pairs = list(zip(nonzero_inds[0], nonzero_inds[1]))  # user-item index 를 zip으로 묶어줘
    random.seed(0)
    num_samples = int(np.ceil(pct_test * len(nonzero_pairs)))  # 샘플 수 반올림
    samples = random.sample(nonzero_pairs, num_samples)  # random하게 샘플링
    user_inds = [index[0] for index in samples]  # 행 정보(user)
    item_inds = [index[1] for index in samples]  # 열 정보(item)
    training_set[user_inds, item_inds] = 0  # 샘플링으로 지정된 애들 값 0으로 바꿔
    training_set.eliminate_zeros()  # 0 없애
    return training_set, test_set, list(set(user_inds))
    # traing_set : pct_test 비율만큼이 Mask처리(0 됨) 된 matrix
    # test_set : Mask 전의 원본 matrix
    # list(set(user_inds) : 어떤 user의 행이 Mask 대상으로 선택되었는지


def implicit_weighted_ALS(training_set, lambda_val=0.1, alpha=40, iterations=10, rank_size=20, seed=0): #ALS 구현 - step3에서 사용
    #  Hu, Koren, Volinsky 이 만든 Implicit weighted ALS 논문 기반으로 함수 구현
    '''
    # parameter 정리
    training_set - user*item 꼴의 트레이닝셋 인풋
    lambda_val - 정규화 위해 사용. 키우면 bias 커지고 variance는 감소함.
    alpha - 논문 상 Cui = 1 + alpha*Rui 부분의 alpha. 값 낮추면 등급간 신뢰도 낮아짐
    iterations - 반복. 늘리면 계산이 늘어나므로, 더욱 수렴하게 됨.
    rank_size - rank size
    seed - seed 지정용
    '''

    # matrix 설정
    conf = (alpha * training_set)
    # sparse 하던 녀석을 dense 하게
    num_user = conf.shape[0]  # user 수 m
    num_item = conf.shape[1]  # item 수 n

    # 설정 된 seed로 random하게 X/Y feature vector 초기화
    rstate = np.random.RandomState(seed)

    X = sparse.csr_matrix(rstate.normal(size=(num_user, rank_size)))  # m x rank 꼴의 랜덤 수
    Y = sparse.csr_matrix(rstate.normal(size=(num_item, rank_size)))  # x n rank 꼴
    # 계산 더 쉽게하기 위해 변경
    X_eye = sparse.eye(num_user)
    Y_eye = sparse.eye(num_item)
    lambda_eye = lambda_val * sparse.eye(rank_size)  # regularization term인 lambda*I.

    # iterations 시작
    for iter_step in range(iterations):
        # 계산시간을 구하기 위해 yTy 와 xTx 를 각 반복문의 시작에서 계산
        yTy = Y.T.dot(Y)
        xTx = X.T.dot(X)
        # Y 고정하고 X 풀기
        for u in range(num_user):
            conf_samp = conf[u, :].toarray()  #user row 가져와서 dense 하게 변환
            pref = conf_samp.copy()
            pref[pref != 0] = 1  # 이진벡터 생성
            CuI = sparse.diags(conf_samp, [0])  # Cu - I term
            yTCuIY = Y.T.dot(CuI).dot(Y)  #  yT(Cu-I)Y term
            yTCupu = Y.T.dot(CuI + Y_eye).dot(pref.T)  # yTCuPu term
            X[u] = spsolve(yTy + yTCuIY + lambda_eye, yTCupu) #Xu = ((yTy + yT(Cu-I)Y + lambda*I)^-1)yTCuPu - 논문 equation 4번식

        # X 고정하고 Y 풀기 - 방식 같으므로 주석 생략
        for i in range(num_item):
            conf_samp = conf[:, i].T.toarray()
            pref = conf_samp.copy()
            pref[pref != 0] = 1
            CiI = sparse.diags(conf_samp, [0])
            xTCiIX = X.T.dot(CiI).dot(X)
            xTCiPi = X.T.dot(CiI + X_eye).dot(pref.T)
            Y[i] = spsolve(xTx + xTCiIX + lambda_eye, xTCiPi)  #Yi = ((xTx + xT(Cu-I)X) + lambda*I)^-1)xTCiPi - 논문 equation 5번식

    # iterations 끝
    return X, Y.T  # feature vectors


##################### step1. DATA 전처리 #####################
#파일 읽어오자
website_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx'
retail_data = pd.read_excel(website_url)


#고객 정보를 확인해보자.
#retail_data.head()
'''
	InvoiceNo	StockCode	Description	Quantity	InvoiceDate	UnitPrice	CustomerID	Country
0	536365	85123A	WHITE HANGING HEART T-LIGHT HOLDER	6	2010-12-01 08:26:00	2.55	17850.0	United Kingdom
1	536365	71053	WHITE METAL LANTERN	6	2010-12-01 08:26:00	3.39	17850.0	United Kingdom
2	536365	84406B	CREAM CUPID HEARTS COAT HANGER	8	2010-12-01 08:26:00	2.75	17850.0	United Kingdom
3	536365	84029G	KNITTED UNION FLAG HOT WATER BOTTLE	6	2010-12-01 08:26:00	3.39	17850.0	United Kingdom
4	536365	84029E	RED WOOLLY HOT
TIE WHITE HEART.	6	2010-12-01 08:26:00	3.39	17850.0	United Kingdom
'''


#데이터 누락된 값 확인
#retail_data.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 541909 entries, 0 to 541908
Data columns (total 8 columns):
InvoiceNo      541909 non-null object
StockCode      541909 non-null object
Description    540455 non-null object
Quantity       541909 non-null int64
InvoiceDate    541909 non-null datetime64[ns]
UnitPrice      541909 non-null float64
CustomerID     406829 non-null float64
Country        541909 non-null object
dtypes: datetime64[ns](1), float64(2), int64(1), object(4)
memory usage: 33.1+ MB
'''
# CustomerID가 누락된게 좀 있음 -> 이런 행들은 삭제 - pd.isnull 활용
cleaned_retail = retail_data.loc[pd.isnull(retail_data.CustomerID) == False]


#이제 데이터 상태 확인
#cleaned_retail.info()
'''
<class 'pandas.core.frame.DataFrame'>
Int64Index: 406829 entries, 0 to 541908
Data columns (total 8 columns):
InvoiceNo      406829 non-null object
StockCode      406829 non-null object
Description    406829 non-null object
Quantity       406829 non-null int64
InvoiceDate    406829 non-null datetime64[ns]
UnitPrice      406829 non-null float64
CustomerID     406829 non-null float64
Country        406829 non-null object
dtypes: datetime64[ns](1), float64(2), int64(1), object(4)
memory usage: 27.9+ MB
'''


#StockCode별  Description을 매치해둔 item_lookup을 만들어두자
item_lookup = cleaned_retail[['StockCode', 'Description']].drop_duplicates()
item_lookup['StockCode'] = item_lookup.StockCode.astype(str)    #형변환
#item_lookup.head()         # 만든 item_lookup 확인
'''
	StockCode	Description
0	85123A	WHITE HANGING HEART T-LIGHT HOLDER
1	71053	WHITE METAL LANTERN
2	84406B	CREAM CUPID HEARTS COAT HANGER
3	84029G	KNITTED UNION FLAG HOT WATER BOTTLE
4	84029E	RED WOOLLY HOTTIE WHITE HEART.
'''


cleaned_retail['CustomerID'] = cleaned_retail.CustomerID.astype(int) # CustomerID 형변환 -> int
cleaned_retail = cleaned_retail[['StockCode', 'Quantity', 'CustomerID']] # 필요한 것만 챙김
grouped_cleaned = cleaned_retail.groupby(['CustomerID', 'StockCode']).sum().reset_index() # 그룹 지어줌 고객 - 상품코드
grouped_cleaned.Quantity.loc[grouped_cleaned.Quantity == 0] = 1 # 구매 합이 0개인 것은 1을 넣어줘서 구매한 것처럼 바꿔줌
grouped_purchased = grouped_cleaned.query('Quantity > 0') # 전체 구매가 양수인 것만 확보
#grouped_purchased.head()
'''
CustomerID	StockCode	Quantity
0	12346	23166	1
1	12347	16008	24
2	12347	17021	36
3	12347	20665	6
4	12347	20719	40
'''


#전처리 마지막 - sparse ratings matrix 만들기
customers = list(np.sort(grouped_purchased.CustomerID.unique())) # Get unique customers
products = list(grouped_purchased.StockCode.unique()) # Get unique products (구매 되었던 애들만)
quantity = list(grouped_purchased.Quantity) # 모든 구매

rows = grouped_purchased.CustomerID.astype('category', categories = customers).cat.codes #행 갯수
cols = grouped_purchased.StockCode.astype('category', categories = products).cat.codes #열 갯수
purchases_sparse = sparse.csr_matrix((quantity, (rows, cols)), shape=(len(customers), len(products)))

#purchases_sparse       # 확인: 4338명 고객, 3664개 item
'''
<4338x3664 sparse matrix of type '<class 'numpy.int64'>'
	with 266723 stored elements in Compressed Sparse Row format>
'''

#sparsity 구해보기
matrix_size = purchases_sparse.shape[0]*purchases_sparse.shape[1] # Number of possible interactions in the matrix
num_purchases = len(purchases_sparse.nonzero()[0]) # Number of items interacted with
sparsity = 100*(1 - (num_purchases/matrix_size))
#sparsity   #확인
'''
98.32190920694744
# 98.32% 의 matrix가 sparse 하므로, 괜찮은 결과를 얻을 수 있음!
# 놀랍게도 collaborative filtering에서는 99.5%까지 sparse여도 ㄱㅊ이래. 해당내용은 CF 찾아보기
'''


################# step2. Training Set과 Test Set 만들기######################
# 정의해 둔 make_train 함수로 Data의 무작위한 퍼센트만큼의 user/item interactions 을 감춰서 Training Set을 만들자
product_train, product_test, product_users_altered = make_train(purchases_sparse, pct_test = 0.2)


########### step3. implicit feedback ALS 구현하기 ###################
user_vecs, item_vecs = implicit_weighted_ALS(product_train, lambda_val = 0.1, alpha = 15, iterations = 1, rank_size = 20)
#user_vecs[0,:].dot(item_vecs).toarray()[0,:5]  # 처음 5개 item에 대한 예시
'''
array([ 0.00644811, -0.0014369 ,  0.00494281,  0.00027502,  0.01275582])
'''

############ step4. ALS 가속화 - 이건 해도 되고 아니어도 뭐 #################
'''
alpha = 15
user_vecs, item_vecs = implicit.alternating_least_squares((product_train*alpha).astype('double'), factors=20,regularization = 0.1,iterations = 50)
'''

########### step5. 추천 System 평가하기 ################
#부터는 다음 시간에 ㅎ