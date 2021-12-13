import torch


def default_gamma(X:torch.Tensor):
    gamma = 1.0 / X.shape[1]
    print(f'Setting default gamma={gamma}')
    return gamma


def rbf_kernel(X:torch.Tensor, gamma:float=None):
    assert len(X.shape) == 2

    if gamma is None:
        gamma = default_gamma(X)
    K = torch.cdist(X, X)
    K.fill_diagonal_(0) # avoid floating point error
    K.pow_(2)
    K.mul_(-gamma)
    K.exp_()
    return K


def local_rbf_kernel(X:torch.Tensor, y:torch.Tensor, gamma:float=None):
    # todo make final representation sparse (optional)
    assert len(X.shape) == 2
    assert len(y.shape) == 1
    assert torch.all(y == y.sort()[0]), 'This function assumes the dataset is sorted by y'

    if gamma is None:
        gamma = default_gamma(X)
    K = torch.zeros((X.shape[0], X.shape[0]))
    y_unique = y.unique()
    for i in range(y_unique[-1] + 1): # compute kernel blockwise for each class
        ind = torch.where(y == y_unique[i])[0]
        start = ind.min()
        end = ind.max() + 1
        K[start:end, start:end] = rbf_kernel(X[start:end, :], gamma=gamma)
    return K


def change_gamma(K:torch.Tensor, old_gamma:float, new_gamma:float):
    assert K.shape[0] == K.shape[1]
    K.log_()
    K.div_(-old_gamma)
    K.mul_(-new_gamma)
    K.exp_()
    return K


if __name__ == "__main__":
    from sklearn.metrics.pairwise import rbf_kernel as rbf_kernel_sklearn
    test_X = torch.rand(100, 128)
    print('Testing default gamma')
    assert torch.allclose(rbf_kernel(test_X), torch.from_numpy(rbf_kernel_sklearn(test_X.numpy())))
    print('Testing gamma=0.026')
    assert torch.allclose(rbf_kernel(test_X, gamma=0.026), torch.from_numpy(rbf_kernel_sklearn(test_X.numpy(), gamma=0.026)))
