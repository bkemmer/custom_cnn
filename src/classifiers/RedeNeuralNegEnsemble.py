import numpy as np
import matplotlib.pyplot as plt

def softmax(A, axis=1):
    A -= np.max(A)
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)

def sigmoid(x):
    return (1/(1+np.exp(-x)))

def correlacao_negativa(yHats, i, lamdba):
    """ Cálculo da correlação negativa

    Args:
        yHats (np.array): Predições das M redes
        y (np.array): Valor (real) de y
        i (int): índice da rede atual
        lamdba (float): termo regularizador do ensamble [0,1]

    Returns:
        np.array: correlação negativa da rede i
    """
    j = np.arange(len(yHats), dtype=int)
    f_n = np.sum(yHats, axis=0)/len(yHats)
    p_i = (yHats[i] - f_n)*np.sum(yHats[j!=i]-f_n, axis=0)
    return lamdba*np.mean(p_i)

def inicializaRede(n, d, h_size, n_classes, seed):  # sourcery skip: merge-dict-assign
    np.random.seed(seed)
    parametros = {}
    parametros['W1'] = np.random.randn(d, h_size)*0.01
    parametros['b1'] = np.zeros((1,h_size))
    parametros['W2'] = np.random.randn(h_size, n_classes)*0.01
    parametros['b2'] = np.zeros((1,n_classes))
    return parametros

def foward_pass(X, parametros, ativacao):
    # forward pass
    W1, b1, W2, b2 = parametros['W1'], parametros['b1'], parametros['W2'], parametros['b2']

    if ativacao == 'sigmoid':
        A = sigmoid(np.dot(X, W1) + b1)
    elif ativacao == 'relu':
        A = np.maximum(0, np.dot(X, W1) + b1)

    y_hat = softmax(np.dot(A, W2) + b2, axis=1)
    return A, y_hat # np.squeeze() squeeze para o caso quando for classificação binária

def backpropagation(yHats, X, y, i, lamdba, A, parametros, ativacao):
    A, W1, b1, W2, b2 = A, parametros['W1'], parametros['b1'], parametros['W2'], parametros['b2']
    N,_ = np.shape(X)

    f_n = np.sum(yHats, axis=0)/len(yHats)
    dJ = (1/N)*((1-lamdba)*(np.squeeze(yHats[i]) - y) + lamdba*(np.squeeze(f_n) - y))
    if len(np.shape(dJ)) < 2:
        dJ = dJ.reshape(-1,1)

    dW2 = A.T @ dJ
    db2 = np.sum(dJ, axis=0, keepdims=True)

    assert(dW2.shape == W2.shape) #, 'dW2 com shape diferente de W2')
    assert(db2.shape == b2.shape) #, 'db2 com shape diferente de b2')

    dA = dJ @ W2.T # derivada do custo
    # derivada parte não-linear
    if ativacao == 'sigmoid':
        dA = dA * (1-A)*A
    elif ativacao == 'relu':
        dA[A<=0] = 0

    dW1 = X.T @ dA # derivada da parte linear
    db1 = np.sum(dA, axis=0, keepdims=True)
    assert(W1.shape == W1.shape) #, 'dW1 com shape diferente de W1')
    assert(db1.shape == b1.shape) #, 'db1 com shape diferente de b1')
    
    return dW1, db1, dW2, db2

def atualizaParametros(parametros, dW1, db1, dW2, db2, taxa_aprendizado):
    parametros['W1'] -= taxa_aprendizado * dW1
    parametros['b1'] -= taxa_aprendizado * db1
    parametros['W2'] -= taxa_aprendizado * dW2
    parametros['b2'] -= taxa_aprendizado * db2

    return parametros

def calculaCusto(yHats, y, lamdba, custo, i, plot):

    shape = np.shape(yHats)
    if len(shape) == 2:
        n_redes, N = shape
    else:
        n_redes, N, n_classes = shape
    # f_n = np.mean(yHats, axis=0)
    custo_redes = []
    for j in range(n_redes):
        custo_redes.append( (1/(2*N))*np.sum((yHats[j] - y)**2) + correlacao_negativa(yHats, j, lamdba) )
    custo.append(np.mean(custo_redes))
    if plot and i % 100 == 0:
        print('{}: {:.4}'.format(i, custo[-1]))
    return custo
 
def redeNeuralSoftmaxEnsemble(X, y, n_redes, lamdba, h_size=100, ativacao='sigmoid', 
                                taxa_aprendizado=0.5, max_iteracoes=10000, custo_min=1e-5, 
                                seed=42, plot=True):
    
    custo = []
    N, d = np.shape(X)
    n_classes = y.shape[1] if len(np.shape(y)) > 1 else 1

    parametros_redes = {}
    for j in range(n_redes):
        parametros_redes[j] = inicializaRede(N, d, h_size, n_classes, seed)
    # matriz contendo as predições das M redes neurais
    
    # Salvando as iterações intermediárias que serão usadas no backpropagation
    if n_classes < 2:
        yHats = np.zeros((n_redes, N))
    else:
        yHats = np.zeros((n_redes, N, n_classes))
    foward_pass_ativacoes = np.zeros((n_redes, N, h_size))
    
    i = 0
    continuar = True
    while continuar:
        
        #forward pass 
        for j in range(n_redes):
            foward_pass_ativacoes[j], yHats[j] = foward_pass(X, parametros_redes[j], ativacao)
        
        # backpropagation
        for j in range(n_redes):
            dW1, db1, dW2, db2 = backpropagation(yHats, X, y, j, lamdba, foward_pass_ativacoes[j], parametros_redes[j], ativacao)
            #atualiza parâmetros
            parametros_redes[j] = atualizaParametros(parametros_redes[j], dW1, db1, dW2, db2, taxa_aprendizado)

        # Cálculo do custo
        custo = calculaCusto(yHats, y, lamdba, custo, i, plot)

        i += 1
        continuar = ((custo[-1] > custo_min) and (i < max_iteracoes))
    # print('Época final: {}\nCusto final: {}'.format(i, custo[-1]))

    return parametros_redes, custo
def mode(x):
    return np.bincount(x).argmax()

def preditorNeuralSoftmaxEnsemble(X_test, parametros, ativacao='sigmoid'): #n_classes, 
    N, _ = np.shape(X_test)
    n_redes = len(parametros)
    yHats = np.zeros((n_redes, N))
    for i, parametros_rede_i in parametros.items():
        W1, b1, W2, b2 = parametros_rede_i['W1'], parametros_rede_i['b1'], parametros_rede_i['W2'], parametros_rede_i['b2']
        if ativacao == 'sigmoid':
            A = sigmoid(np.dot(X_test, W1) + b1)
        elif ativacao == 'relu':
            A = np.maximum(0, np.dot(X_test, W1) + b1)
        Z = np.dot(A, W2) + b2
        yHats[i] = np.argmax(Z, axis=1)
    return np.apply_along_axis(mode, 1, yHats.T.astype(int))

def teste():
    from utils import Ymulticlasse
    N = 100 # number of points per class
    D = 2 # dimensionality
    K = 3 # number of classes
    X = np.zeros((N*K,D)) # data matrix (each row = single example)
    y = np.zeros(N*K, dtype='uint8') # class labels
    for j in range(K):
        ix = range(N*j,N*(j+1))
        r = np.linspace(0.0,1,N) # radius
        t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j
    y_old = np.copy(y)
    
    y, _ = Ymulticlasse(y, y)
    print("Teste sigmoid:\n")
    parametros, custo = redeNeuralSoftmaxEnsemble(X, y, n_redes=4, lamdba=.5, h_size=100, ativacao='sigmoid', taxa_aprendizado=1, max_iteracoes=5000, custo_min=1e-5, plot=True)
    y_hat = preditorNeuralSoftmaxEnsemble(X, parametros, ativacao='sigmoid') #K, 
    print('training accuracy: %.2f' % (np.mean(y_old == y_hat)))
    print("\nTeste relu:\n")
    parametros, custo = redeNeuralSoftmaxEnsemble(X, y, n_redes=4, lamdba=.5, h_size=100, ativacao='relu', taxa_aprendizado=1, max_iteracoes=10000, custo_min=1e-5, plot=True)
    y_hat = preditorNeuralSoftmaxEnsemble(X, parametros, ativacao='relu')
    print('training accuracy: %.2f' % (np.mean(y_old == y_hat)))

if __name__ == '__main__':
    teste()
