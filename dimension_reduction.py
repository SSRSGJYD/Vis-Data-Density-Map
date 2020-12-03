import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import manifold, decomposition
import time


if __name__ == "__main__":
    X = np.load('data/sampled_image.npy')
    X = X.reshape(len(X), -1)
    y = np.load('data/sampled_label.npy')

    s = time.time()
    X = decomposition.TruncatedSVD(n_components=50).fit_transform(X)
    e = time.time()
    print('PCA:', e-s)

    for perplexity in [30, 50]:
        for niter in [2000, 3500, 5000]:
            s = time.time()
            tsne = manifold.TSNE(perplexity=perplexity, n_iter=niter, n_components=2, init='pca', random_state=0)
            X_tsne = tsne.fit_transform(X)
            e = time.time()
            print(niter, e-s)
            np.save('./tsne/pca50_{}_{}.npy'.format(perplexity, niter), X_tsne)
            plt.scatter(X_tsne[:,0], X_tsne[:,1], s=10, c=y)
            plt.savefig('./tsne/pca50_{}_{}.png'.format(perplexity, niter))
            plt.cla()
            # with open('./tsne.json', 'w') as f:
            #     import json
            #     X_tsne = X_tsne.astype(np.float64)
            #     json.dump({'data': X_tsne.astype(np.float64).tolist(), 
            #                 'label': y.tolist(),
            #                 'xmin':np.min(X_tsne[:, 0]),
            #                 'xmax':np.max(X_tsne[:, 0]),
            #                 'ymin':np.min(X_tsne[:, 1]),
            #                 'ymax':np.max(X_tsne[:, 1])}, f)