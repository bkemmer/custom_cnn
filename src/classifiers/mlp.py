import numpy as np
from scipy.special import softmax
from sklearn.metrics import accuracy_score


class multilayer_perceptron:

    '''
        input layer - hidden layer: f(a * x) = sigmoid
        hidden layer - output layer: g (b * f(a * x)) = softmax
        loss function = cross-entropy
    '''
    def __init__(self, num_hidden):
        self.n_hidden = num_hidden # neurons number hidden layer
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


    # Foward step
    def forward(self, X, a, bias_a, b, bias_b):
        z_in = np.matmul(X, a.T) + bias_a.T # X(n, m) * a.T(m, num_hid)
        z_i = self.sigmoid(z_in) 

        y_in = np.matmul(z_i, b.T) + bias_b.T # z_i (n, num_hid) * b.T (num_hid, num_class)
        y_i = softmax(y_in, axis=1) # (n, num_class)
        return z_in, z_i, y_in, y_i

    def predict_(self, x_i):
        _, _, _, y_i = self.forward(X=x_i, a=self.a, bias_a=self.bias_a, b=self.b, bias_b=self.bias_b)
        return y_i

    def predict(self, X):
        y_ = np.concatenate([self.predict_(x_i) for x_i in X])
        return self.map_y(np.asarray(y_))

    # Train neural network
    def fit(self, X, y_d, X_val = None, y_val_d = None,alfa=0.1, n_max=10000, error_min = 10**-5):
        n, m = X.shape
        n_c = y_d.shape[1]

        a = np.random.normal(scale=1 / m ** .5, size=(self.n_hidden, m))
        bias_a = np.random.normal(scale=1 / m ** .5, size=(self.n_hidden, 1))
        print('A: ', a.shape)
        print('bias_A: ', bias_a.shape)
        b = np.random.normal(scale=1 / m ** .5, size=(n_c, self.n_hidden))
        bias_b = np.random.normal(scale=1 / m ** .5, size=(n_c, 1))
        print('B: ', b.shape)
        print('bias_B: ', bias_b.shape)



        '''
            Apenas se tiver X_val e y_val_d
        '''
        if X_val is not None and y_val_d is not None:
            acc_val_max = 0
            bckp_a = None
            bckp_bias_a = None
            bckp_b = None
            bckp_bias_b = None

        error_list = []

        for epoch in range(n_max):
            sum_d_B = 0; sum_d_bias_B = 0; sum_d_A = 0; sum_d_bias_A = 0; sum_error = 0
            for index in range(n):
                # Update weights in Batch process way (Batelada)
                x_i = X[[index]]
                y_d_i = y_d[index]
                # forward
                z_in, z_i, y_in, y_i = self.forward(x_i, a, bias_a, b, bias_b)

                # backward
                '''
                   update B
                   Note by chain rule: w gradient  = loss gradient * Activation gradient * h gradient
                   In our example, we will be using softmax activation at the output layer. Loss will be computed by using the Cross Entropy Loss formula.
                   This derivative reduces to y*-y, which is the prediction, y*, subtract the expected value, y.
                   = (y-y_d) * h gradient => error * Z_i => np.matmul(d_y_i.T, z_i)
               '''
                error = np.sum(np.abs(y_i - y_d_i))
                d_y_i = y_i - y_d_i # loss + output gradient
                d_B = np.matmul(d_y_i.T, z_i)
                d_bias_B = d_y_i.T
                '''
                    update A = derivative error from Zi (dEt/dZi) * derivative zi from ai (dZi/dAi)
                    dEt/dZi = 1* g(Yin(n)*b
                    
                    dZi/dAi = dZi/dZin * dZin/daij
                        dZi/dZin = f(Zin)
                        dZin/daij = X
                    dZi/dAi = f(Zin) * X
                    = dEt/dZi * dZi/dAi 
                '''
                dEt_dZi = np.matmul(d_y_i, b)
                dZi_dZin = self.gradiente_sigmoid(y=z_in)
                dZin_dA = x_i

                d_A = np.matmul((dEt_dZi * dZi_dZin).T, dZin_dA)
                d_bias_A = (dEt_dZi * dZi_dZin).T

                sum_d_B += d_B
                sum_d_bias_B += d_bias_B
                sum_d_A += d_A
                sum_d_bias_A += d_bias_A
                sum_error += error

            error_list.append(1/n * sum_error)
           
            #print('sum_d_B: ', (1/n * np.sum(np.abs(sum_d_bias_B))), ' - sum_d_bias_B: ', (1/n * np.sum(np.abs(sum_d_bias_B))), ' - sum_d_A: ', np.sum(np.abs((1/n * sum_d_A))), ' - sum_d_bias_A:', np.sum(np.abs((1/n * sum_d_bias_A))))
            b = b - (alfa * (1/n * sum_d_B))
            bias_b = bias_b - (alfa * (1/n * sum_d_bias_B))
            a = a - (alfa * (1/n * sum_d_A))
            bias_a = bias_a - (alfa * (1/n * sum_d_bias_A))
            
            '''
                Se tiver conjunto de validação: salva apenas o peso da epoca atual se reduzir o erro de validação, caso contrário:
                    - continua com o peso salvo numa epoca anterior
            '''
            if X_val is not None and y_val_d is not None:
                y_val_pred = np.concatenate([self.forward(X=x_val_i, a=a, bias_a=bias_a, b=b, bias_b=bias_b)[-1] for x_val_i in X_val])
                y_val_pred = self.map_y(np.asarray(y_val_pred))
                acc_ = accuracy_score(y_true=y_val_d, y_pred=y_val_pred)
                #print("Epoch {}:".format(epoch), "error= {}".format(1/n * sum_error), 'acc= {}'.format(acc_), 'acc_max= {}'.format(acc_val_max))
                if acc_ > acc_val_max:
                    acc_val_max = acc_
                    bckp_a = a
                    bckp_bias_a = bias_a
                    bckp_b = b
                    bckp_bias_b = bias_b

        if X_val is not None and y_val_d is not None:
            self.a = bckp_a; self.b = bckp_b; self.bias_a = bckp_bias_a; self.bias_b = bckp_bias_b
        else:
            self.a = a; self.b = b; self.bias_a = bias_a; self.bias_b = bias_b
            
        return self.b, self.bias_b, self.a, self.bias_a, error_list





