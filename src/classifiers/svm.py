#references:  https://sandipanweb.wordpress.com/2018/04/23/implementing-a-soft-margin-kernelized-support-vector-machine-binary-classifier-with-quadratic-programming-in-r-and-python/
import numpy as np
import cvxopt
from scipy import stats

class svm:

    def svm_kernel_fit(self, X, kernel_type, degree = None):
        # todas as observações de treino
        if kernel_type == 'linear':
            return np.matmul(X, X.T)
        elif kernel_type == 'polynomial':
            return (np.matmul(X, X.T)+1)**degree


    def svm_kernel_predict(self, X, x, kernel_type, degree = None):
        # observação com SV
        if kernel_type == 'linear':
            return np.matmul(X, x.T)
        elif kernel_type == 'polynomial':
            return (np.matmul(X, x.T)+1)**degree
        
    def fit(self, X, y, kernel_type='linear', degree=None, C=None):
        n_samples, n_features = X.shape
    
        K = self.svm_kernel_fit(X, kernel_type, degree)
        P = cvxopt.matrix(np.multiply(np.matmul(y, y.T), K).astype('float'))
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        # A = cvxopt.matrix(y.T.astype('float'))
        A = cvxopt.matrix(y.reshape((1,n_samples)), tc='d')
        b = cvxopt.matrix(0.0)
        
        if C is None or C==0:      # hard-margin SVM
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:              # soft-margin SVM
            G = cvxopt.matrix(np.vstack((np.diag(np.ones(n_samples) * -1), np.identity(n_samples))))
            h = cvxopt.matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * C)))    
            
        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        # Lagrange multipliers
        alphas = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers
        sv_index = alphas > 1e-5
        ind = np.arange(len(alphas))[sv_index]
        self.alphas_sv = alphas[sv_index]
        self.alphas_sv = np.expand_dims(self.alphas_sv, axis=1)
        self.X_sv = X[sv_index]
        self.y_sv = y[sv_index]
        
        # b
        self.b = 0
        for i in range(len(self.X_sv)):
            b_ = self.y_sv[i] - np.sum(np.multiply(self.alphas_sv, self.y_sv).T * K[ind[i],sv_index])
            self.b = self.b + b_
        
        self.b = self.b/len(self.X_sv)

        return alphas
        

    def predict(self, X):
        y_pred = []
        for index in range(0, X.shape[0]):
            xi = X[[index]]
            # calculo de Kernel para cada xi em relacao ao Support Vector
            K_X_xi = self.svm_kernel_predict(X=self.X_sv, x=xi, kernel_type='polynomial', degree=2)
        
            #predicao de cada x_i
            y_pred_i = np.sum(np.multiply(np.multiply(self.alphas_sv, self.y_sv), K_X_xi))
            y_pred.append(np.sign(y_pred_i + self.b)) 
        return np.asarray(y_pred)



class svm_multiclass:

    

    def fit(self, X, y, kernel_type='linear', degree=None, C=None, strategy='one-vs-one'):
        '''
            y = one-of-n encoding
        '''
        self.clf_svm = svm()
        n, nc = y.shape
        print('Labels number: ', nc)
        labels = np.unique(y, axis=0)
        print('Labels: ', labels)
        self.degree = degree
        self.C = C
        self.kernel_type = kernel_type
        

        if strategy=='one-vs-one':
            self.classifiers = []
            for index_class_i in range(0, nc):
                for index_class_j in range(index_class_i +1, nc):
                    print('index i: ', index_class_i, ' index j: ', index_class_j)
                    d = {}
                    d['classifier_A'] = labels[index_class_i]
                    d['classifier_B'] = labels[index_class_j]
                    d['instances_classifier_A'] = [index for index, i in enumerate(y) if (i==labels[index_class_i]).all()]
                    d['instances_classifier_B'] = [index for index, i in enumerate(y) if (i==labels[index_class_j]).all()]
                    
                    X_ = np.concatenate((X[d['instances_classifier_A']], X[d['instances_classifier_B']]), axis=0)
                    print('X_: ', X_.shape)
                    y_ = np.concatenate((y[d['instances_classifier_A']], y[d['instances_classifier_B']]), axis=0)
                    print('y_: ', y_.shape)
                    
                    # map one-of-n encoding to 1 and -1
                    # classifier_A => 1
                    # classifier_B => -1
                    y_ = np.asarray([np.asarray([1]) if (label_==d['classifier_A']).all() else np.asarray([-1]) if (label_==d['classifier_B']).all() else 'error' for label_ in y_])
                    print('y_: ', y_.shape)
                    d['X_'] = X_
                    d['y_'] = y_
                    
                    alphas = self.clf_svm.fit(X=X_, y=y_, kernel_type=self.kernel_type, degree=self.degree, C=self.C)
                    d['X_sv'] = self.clf_svm.X_sv
                    d['y_sv'] = self.clf_svm.y_sv
                    d['alphas_sv'] = self.clf_svm.alphas_sv
                    d['b'] = self.clf_svm.b
                    d['alphas'] = alphas
                    
                    self.classifiers.append(d)

        return self.classifiers


    def predict_(self, X_sv, xi, alphas_sv, y_sv, b):
        # calculo de Kernel para cada xi em relacao ao Support Vector
        K_X_xi = self.clf_svm.svm_kernel_predict(X=X_sv, x=xi, kernel_type=self.kernel_type, degree=self.degree)
        #predicao de cada x_i
        y_pred_i = np.sum(np.multiply(np.multiply(alphas_sv, y_sv), K_X_xi))
        y_pred_i = np.sign(y_pred_i + b)
        return np.asarray(y_pred_i)


    def predict(self, X):
        y_pred = []
        y_pred_classifiers = []
        print()
        for index in range(0, X.shape[0]):
            # instance xi
            xi = X[[index]]

            aux = []
            for classifier in self.classifiers:
                # label mapped: 1 or -1
                pred_label_mapped = self.predict_(X_sv=classifier['X_sv'], xi=xi, alphas_sv= classifier['alphas_sv'], y_sv=classifier['y_sv'], b=classifier['b'])
                # necessario voltar para o encoding original para apendar no y_pred_classifier
                aux.append(np.asarray(classifier['classifier_A'] if pred_label_mapped==1 else classifier['classifier_B']))
                  
            # all predictions
            y_pred_classifiers.append(aux)
            # majoritary class
            y_pred.append(stats.mode(aux).mode[0])
        return y_pred, y_pred_classifiers
            
        
            
