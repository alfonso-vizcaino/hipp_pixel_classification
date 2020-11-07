from tensorflow.keras.initializers import Initializer
from sklearn.cluster import KMeans


class InitCentersKMeans(Initializer):
    """ Initializer for initialization of centers of RBF network
        by clustering the given data set.
    # Arguments
        X: matrix, dataset
    """

    def __init__(self, X, max_iter=100, use_shape=True, k=19 ):
        self.X = X
        self.max_iter = max_iter
        self.use_shape = use_shape
        self.k = k
        
        super().__init__()

    def __call__(self, shape, dtype=None):
        assert shape[1:] == self.X.shape[1:]

        if self.use_shape :
            n_centers = shape[0]
        else :
            n_centers = self.k
            
        print("n_centers=",n_centers)
        km = KMeans(n_clusters=n_centers, max_iter=self.max_iter, verbose=0)

        #dMax = max([np.abs(c1 - c2) for c1 in self.centers for c2 in self.centers])
        #self.stds = np.repeat(dMax / np.sqrt(2*self.k), self.k)
        #print(self.stds)
        
        km.fit(self.X)
        return km.cluster_centers_
