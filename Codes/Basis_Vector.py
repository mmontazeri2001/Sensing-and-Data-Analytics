def Basis_Vector (eigenvec,delta):
    import numpy as np
    from scipy.spatial import distance
    B_V = np.array([eigenvec[:,0]])
    n = 0
    for i in range(1,eigenvec.shape[1]):
        check = np.array([eigenvec[:,i]])
        concat = np.concatenate((B_V,check))
        dist = distance.pdist(concat,'euclidean')
        if (dist > delta) and (i<201):
            B_V = B_V + (0.0001)* (B_V - check)
        elif (i>201) :
            break
    return (B_V)
