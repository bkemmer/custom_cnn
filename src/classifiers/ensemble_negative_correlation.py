import numpy as np
from scipy.special import softmax
from sklearn.metrics import accuracy_score


class ensemble:

    '''
        input layer - hidden layer: f(a * x) = sigmoid
        hidden layer - output layer: g (b * f(a * x)) = sigmoid
        loss function = 
    '''
    def __init__(self, num_hidden = 10, number_classifiers = 4):
        self.n_hidden = num_hidden # neurons number hidden layer
        self.number_classifiers = number_classifiers
        self.a = None
        self.b = None
        self.bias_a = None
        self.bias_b = None
        

    # map prob output vector
    def map_y(self, y):
        aux = np.zeros(y.shape, dtype=int)
        for index, max_ in enumerate([np.argmax(y_i)for y_i in y]):
            aux[index][max_] = 1
        return aux

    # activation function - hidden layer
    def sigmoid(self, z_in):
        return 1/(1 + np.exp(-z_in))

    # sigmoid Activation function derivative 
    def gradiente_sigmoid(self, y): # derivative of sigmoid
        #y *(1 - y)
        return np.multiply(self.sigmoid(y), (np.subtract(1, self.sigmoid(y))))

    def gradient_error(self, f_i, f_n, lamdba, y_d_i):
        return (1-lamdba) * (f_i - y_d_i) + lamdba * (f_n - y_d_i)

    # Foward step
    def forward(self, X, a, bias_a, b, bias_b):
        z_in = np.matmul(X, a.T) + bias_a.T # X(n, m) * a.T(m, num_hid)
        z_i = self.sigmoid(z_in) 

        y_in = np.matmul(z_i, b.T) + bias_b.T # z_i (n, num_hid) * b.T (num_hid, num_class)
        # y_i = softmax(y_in, axis=1) # (n, num_class)
        y_i = self.sigmoid(y_in) # (n, num_class)
        return z_in, z_i, y_in, y_i

    def predict_(self, x_i):
        y_i = [None] * self.number_classifiers
        for index in range(0, self.number_classifiers):
            _, _, _, y_i[index] = self.forward(x_i, self.a[index], self.bias_a[index], self.b[index], self.bias_b[index])

        f_n = np.mean(y_i, axis=0)  # output mean: shape = nc

        return f_n

    def predict(self, X):
        y_ = np.concatenate([self.predict_(x_i) for x_i in X])
        return self.map_y(np.asarray(y_))

    # Train neural network
    def fit(self, X, y_d, lamdba = 1, alfa=0.1, n_max=10000):
        n, m = X.shape
        n_c = y_d.shape[1]
        number_classifiers = self.number_classifiers

        #shuffle
        arr = np.arange(n)
        np.random.shuffle(arr)
        X = X[arr]
        y_d = y_d[arr]



        z_in = [None] * number_classifiers; z_i = [None] * number_classifiers; y_in = [None] * number_classifiers; y_i = [None] * number_classifiers
        p_i = [None] * number_classifiers; error_i = [None] * number_classifiers
        grad_error_i = [None] * number_classifiers; grad_output = [None] * number_classifiers; grad_y_in = [None] * number_classifiers; d_B = [None] * number_classifiers; d_bias_B = [None] * number_classifiers
        dEt_dZi = [None] * number_classifiers; dZi_dZin = [None] * number_classifiers; dZin_dA = [None] * number_classifiers; d_A = [None] * number_classifiers; d_bias_A = [None] * number_classifiers
        a = []; bias_a = []; b = []; bias_b = []
        for index in range(0, number_classifiers):    
            a.append(np.random.normal(scale=1 / m ** .5, size=(self.n_hidden, m))) 
            bias_a.append(np.random.normal(scale=1 / m ** .5, size=(self.n_hidden, 1))) 
            b.append(np.random.normal(scale=1 / m ** .5, size=(n_c, self.n_hidden)))
            bias_b.append(np.random.normal(scale=1 / m ** .5, size=(n_c, 1)))


        # train ensemble
        epoch_abs_error_list = []
        epoch_acc = []
        for epoch in range(n_max):
            epoch_abs_error = 0
            acc_ = 0
            for instance in range(n):
                # Update weights in pattern-pattern process way
                x_i = X[[instance]]
                y_d_i = y_d[instance]
                # forward
                for index in range(0, number_classifiers): 
                    z_in[index], z_i[index], y_in[index], y_i[index] = self.forward(x_i, a[index], bias_a[index], b[index], bias_b[index])

                # calculate p_i
                # Paper: f_i = y_i
                f_n = np.mean(y_i, axis=0) # output mean: shape = nc
                epoch_abs_error += np.sum(np.abs(f_n - y_d_i))
                if np.argmax(f_n) == np.argmax(y_d_i):
                    acc_ += 1

                for i in range(0, number_classifiers):
                    sum_pred_j = 0
                    for j in range(0, number_classifiers):
                        #p_i = (pred_i - pred) * \sum j != i (pred_j - pred)
                        if i != j:
                            sum_pred_j += (y_i[j] - f_n)
                    p_i[i] = (y_i[i] - f_n) * sum_pred_j
                    # error_i = 1/n * \sum * E_i_n
                    # = 1/n * \sum * 1/2 * (F_i_n - dn)^2 + 1/n * \sum lambda pi_n
                    error_i[i] = 1/2 * (y_i[i] - y_d_i)**2 + p_i[i] * lamdba

                # backward
                for index in range(0, number_classifiers):
                    '''
                       update B
                       Note by chain rule: w gradient  = loss gradient * Activation gradient * h gradient
                       = derivative error * derivative g(yink) * derivative y_in
                   '''
                    grad_error_i[index] = self.gradient_error(f_i= y_i[index], f_n=f_n, lamdba=lamdba, y_d_i=y_d_i)
                    grad_output[index] = self.gradiente_sigmoid(y=y_in[index])
                    grad_y_in[index] = z_i[index] # n x n_hidden
                    d_B[index] = np.matmul(np.multiply(grad_error_i[index], grad_output[index]).T, grad_y_in[index]) # (n x nc).T * n x n_hidden
                    d_bias_B[index] = np.multiply(grad_error_i[index], grad_output[index]).T
                    b = b - (alfa * d_B[index])
                    bias_b = bias_b - (alfa * d_bias_B[index])

                    '''
                        update A = derivative error from Zi (dEt/dZi) * derivative zi from ai (dZi/dAi)
                        = (derivative error * der_g(Yin(n) * b) * der_f(zin) * x_i
                    '''
                    dEt_dZi[index] = np.matmul(np.multiply(grad_error_i[index], grad_output[index]), b[index])
                    dZi_dZin[index] = self.gradiente_sigmoid(y=z_in[index])
                    dZin_dA = x_i
                    d_A[index] = np.matmul((dEt_dZi[index] * dZi_dZin[index]).T, dZin_dA)
                    d_bias_A[index] = (dEt_dZi[index] * dZi_dZin[index]).T
                    a = a - (alfa * d_A[index])
                    bias_a = bias_a - (alfa * d_bias_A[index])

            epoch_abs_error_mean = 1/n * epoch_abs_error
            epoch_abs_error_list.append(epoch_abs_error_mean)
            acc_mean = 1/n * acc_
            epoch_acc.append(acc_mean)
            #print('Epoch: ', epoch, ' - MAE: ', epoch_abs_error_mean, ' - pattern-pattern acc: ', acc_mean)


        self.b = b; self.bias_b = bias_b; self.a = a; self.bias_a = bias_a
        y_train_pred = self.predict(X=X)
        #print(accuracy_score(y_pred=y_train_pred, y_true=y_d))

        return self.b, self.bias_b, self.a, self.bias_a, epoch_abs_error_list, epoch_acc





