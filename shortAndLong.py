import numpy as np
from scipy.optimize import minimize

def shortAndLong(size, sr_constraint, lr_constraint, F1):
    M = np.identity(size) # initialize M
    omega = np.identity(size)

    # Flatten M and omega for scipy.optimize
    m = M.T.reshape(1,-1)[0]

    # Get 1-indexed column of a flattened matrix.
    def get_col(x, col):
        return x[(col-1) * size : col * size]

    def objective_func(x):
        # The Forbenius norm of (MM'-Omega)
        recovered = x.reshape((size, size)).T
        return np.linalg.norm(np.dot(recovered, recovered.T)-omega, 'fro')

    def sr_constraint_func(x, sr_cons):
        return x[size*(sr_cons[1]-1) + sr_cons[0]-1]
    def lr_constraint_func(x, lr_cons):
        return np.dot(F1[lr_cons[0]-1, :], get_col(x, lr_cons[1]))
    cons = []
    for sr_cons in sr_constraint:
        cons.append({'type':'eq', 'fun':sr_constraint_func, 'args':(sr_cons,)})
    for lr_cons in lr_constraint:
        cons.append({'type':'eq', 'fun':lr_constraint_func, 'args':(lr_cons,)})
    
    result = minimize(objective_func, M, constraints = cons)
    M = result.x.reshape((size, size)).T
    print(M)
    print(result.message)

    return M

def test():
    size = 3
    F1 = np.array([[1,2,3],[4,5,6],[7,8,9]])
    sr_constraint = [(1,1)]
    lr_constraint = [(1,2), (3,1)]
    shortAndLong(size, sr_constraint, lr_constraint, F1)

if __name__ == "__main__":
    test()