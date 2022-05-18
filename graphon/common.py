import numpy as np

def fully_connected_graph(m):
    return np.ones((m,m))

def erdos_renyi(m,p=0.8):
    assert 0 <= p <= 1
    return np.ones((m,m)) * p

def stochastic_block(m,p=0.9,q=0.4):
    w = np.zeros((m,m))
    for a in range(m):
        for b in range(m):
            if a<m/2 and b<m/2:
                w[a][b] = p
            elif a>=m/2 and b>=m/2:
                w[a][b] = p
            else:
                w[a][b] = q
    
    return w

def random_geometric(m):
    w = np.zeros((m,m))
    for a in range(m):
        for b in range(m):
            w[a][b] = min(b/m-a/m, 1-b/m+a/m)
            
    return w

    