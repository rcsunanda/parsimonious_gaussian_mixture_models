import numpy as np
from sklearn import mixture
from sklearn import metrics

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib as mpl
import itertools

def compute_rand_index(cluster1, cluster2):
    n = len(cluster1)
    total_pairs = n * (n-1) / 2
    pairs_in_agreement = 0
    # total_iter = 0

    for i in range(n):
        for j in range(i+1, n):
            # total_iter = total_iter + 1
            if (cluster1[i] == cluster1 [j] and cluster2[i] == cluster2[j]) or  \
                (cluster1[i] != cluster1 [j] and cluster2[i] != cluster2[j]):
                pairs_in_agreement = pairs_in_agreement + 1

    rand_index = pairs_in_agreement / total_pairs
    # print("total_pairs={}, total_iter={}, pairs_in_agreement={}, rand_index={}".
    #       format(total_pairs, total_iter, pairs_in_agreement, rand_index))
    return rand_index


def make_ellipses(gmm, ax, X_data):
    color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue', 'darkorange'])
    scatter_markers = itertools.cycle(['o', '+', 'v', '^'])
    for n, (mean, cov, color, marker) in enumerate(zip(gmm.means_, gmm.covariances_,
                                                       color_iter, scatter_markers)):
        covariances = gmm.covariances_[n][:2, :2]
        # covariances = gmm.covariances_
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        # v = v * 3   # We do this to make ellipse bigger
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = mpl.patches.Ellipse(xy=gmm.means_[n, :2], width=v[0], height=v[1],
                                  angle=180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
        # print(marker)
        plt.scatter(X_data[:, 0], X_data[:, 1], s=1, marker=marker)


def plot_score_contours(gmm, ax, X_data):
    x = np.linspace(-20., 30.)
    y = np.linspace(-20., 40.)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = -gmm.score_samples(XX)
    Z = Z.reshape(X.shape)

    CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
                     levels=np.logspace(0, 3, 10))
    CB = plt.colorbar(CS, shrink=0.8, extend='both')
    plt.scatter(X_data[:, 0], X_data[:, 1], .8)


n_samples = 300  # was 300 originally

# generate random sample, two components
np.random.seed(0)

# generate spherical data centered on (20, 20)
mean1 = [10, 10]
shifted_gaussian = np.random.randn(n_samples, 2) + np.array(mean1)
class1_target = np.zeros((n_samples, 1))

# generate zero centered stretched Gaussian data
mean2 = [0, 0]
C = np.array([[0., -0.7], [3.5, .7]])
stretched_gaussian = np.dot(np.random.randn(n_samples, 2), C) + mean2
class2_target = np.ones((n_samples, 1))

# concatenate the two datasets into the final training set
X_train = np.vstack([shifted_gaussian, stretched_gaussian])
y_train = np.vstack([class1_target, class2_target])

# fit a Gaussian Mixture Model with two components
gmm = mixture.GaussianMixture(n_components=2, covariance_type='full')
gmm.fit(X_train)

# print some model selection metrics
bic = gmm.bic(X_train)    # Bayesian Information Criterion
aic = gmm.aic(X_train)    # Akaike Information Criterion
print("Model selection metrics (the lower the better): BIC={}, AIC={}".format(bic, aic))

# predict on training data and measure training Rand index and adjusted Rand index
pred_train = gmm.predict(X_train)

rand_index = compute_rand_index(y_train, pred_train)
adjusted_rand_index = metrics.adjusted_rand_score(y_train.flatten(), pred_train.flatten())
print("Training data: Rand index={}, Adjusted Rand index={}".format(rand_index, adjusted_rand_index))

# Generate some test data
test_shifted_gaussian = np.random.randn(n_samples, 2) + np.array(mean1)
test_class1_target = np.zeros((n_samples, 1))
test_stretched_gaussian = np.dot(np.random.randn(n_samples, 2), C) + mean2
test_class2_target = np.ones((n_samples, 1))
# concatenate the two datasets into the final test set
X_test = np.vstack([test_shifted_gaussian, test_stretched_gaussian])
y_test = np.vstack([test_class1_target, test_class2_target])


# predict on test data and measure training Rand index and adjusted Rand index
pred_test = gmm.predict(X_test)

test_rand_index = compute_rand_index(y_test, pred_test)
test_adjusted_rand_index = metrics.adjusted_rand_score(y_test.flatten(), pred_test.flatten())
print("Test data: Rand index={}, Adjusted Rand index={}".format(test_rand_index, test_adjusted_rand_index))


# display predicted scores by the model as a contour plot
# plot_score_contours(gmm, None, X_train)
# plt.title('Negative log-likelihood predicted by a GMM')
# plt.axis('tight')


# Plot an ellipse for each Gaussian component
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim([-10, 15])
ax.set_ylim([-5, 15])
make_ellipses(gmm, ax, X_train)

plt.show()
