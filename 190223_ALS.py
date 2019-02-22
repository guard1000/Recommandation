import numpy as np
class ALS():
    def __init__(self, R, f_dim, reg=0.3, max_iter=100, verbose=True, trace=True):

        '''
        R: a number of users by a number of items.
           Each col vec of R is user vector (all item score for an user)
           Each row vec of R is item vector (all score by user for an item)
        f_dim: a dimension of the feature space
        '''

        self.R = R
        self.f_dim = f_dim
        n_user, n_item = R.shape
        self.X = np.random.normal(size=[n_user, f_dim])
        self.Y = np.random.normal(size=[n_item, f_dim])
        self.reg = reg
        self.max_iter = max_iter
        self.verbose = verbose
        self.trace = trace
        self.thres = 0.005

    def __log(self, msg):
        if self.verbose:
            print(msg)

    def __iterate(self, A, user=True):
        '''
        A: fixed matrix
        A: n by f_dim matrix where n is a number of (user, item)
        return: n by f_dim matrix
        '''
        B = self.R.T.todense() if user else self.R.todense()
        ATA = np.dot(A.T, A)
        lambdaI = self.reg * np.eye(self.f_dim)
        ATB = np.dot(A.T, B)
        return np.transpose(np.dot(np.linalg.inv(ATA + lambdaI), ATB))

    def train(self):
        self.__log('start to train')
        prev_e = np.inf
        for i in range(self.max_iter):
            self.X = self.__iterate(self.Y, user=True)
            self.Y = self.__iterate(self.X, user=False)
            if self.trace:
                e = self.rmse(self.R)
                self.__log('%d/%d, %s' % (i+1, self.max_iter, str(e)))
                if 1 - (e / prev_e) < self.thres:
                    self.__log('reach to threshold, stop iteration')
                    break
                prev_e = e
            else:
                self.__log('%d/%d' % (i+1, self.max_iter))

    def rmse(self, R):
        nonzeros = R.nonzero()
        n_element = nonzeros[0].size
        rmse, num = 0, 0
        my_R = self.todense()
        for i in iter(range(n_element)):
            idx = (nonzeros[0][i], nonzeros[1][i])
            rmse += (R[idx] - my_R[idx]) ** 2
            num += 1
        return np.sqrt(rmse / num)

    def todense(self):
        return np.dot(self.X, self.Y.T)