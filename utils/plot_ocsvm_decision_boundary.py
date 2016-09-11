import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from scipy import stats
from sklearn import svm


def plot_ocsvm_decision_boundary(ax=None, X=None, y=None, clf=None):
    if ax is None:
        ax = plt.gca()
    
    assert X.shape[1] == 2
    
    # Calculate x and y limits
    xmin, ymin = np.min(X, axis=0)
    xmax, ymax = np.max(X, axis=0)
    
    xrange = xmax-xmin
    yrange = ymax-ymin
    
    xlims = xmin-xrange*0.1, xmax+xrange*0.1
    ylims = ymin-yrange*0.1, ymax+yrange*0.1
    
    xx, yy = np.meshgrid(np.linspace(xlims[0], xlims[1], 1000), np.linspace(ylims[0], ylims[1], 1000))
    
    pos_class_fraction = np.mean(y)
    y_pred = clf.decision_function(X).ravel()
    #print y_pred
    threshold = stats.scoreatpercentile(y_pred, 100 * (1-pos_class_fraction))
    #print threshold
    #print sum(y_pred > threshold)
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7),
                     cmap=plt.cm.Blues_r)
    a = ax.contour(xx, yy, Z, levels=[threshold],
                        linewidths=2, colors='red')
    ax.contourf(xx, yy, Z, levels=[threshold, Z.max()],
                     colors='orange')
    b = ax.scatter(X[np.logical_not(y), 0], X[np.logical_not(y), 1], c='white')
    c = ax.scatter(X[y.astype('bool'), 0], X[y.astype('bool'), 1], c='black')
    ax.axis('tight')
    ax.legend(
        [a.collections[0], b, c],
        ['learned decision function', 'true inliers', 'true outliers'],
        prop=matplotlib.font_manager.FontProperties(size=11))
    ax.set_xlim(xlims[0], xlims[1])
    ax.set_ylim(ylims[0], ylims[1])
    
    
    
if __name__ == '__main__':
    # Example settings
    n_samples = 200
    outliers_fraction = 0.25

    ocsvm = svm.OneClassSVM(nu=0.95*outliers_fraction + 0.05, kernel="rbf", gamma=0.9)
   
    n_inliers = np.rint((1. - outliers_fraction) * n_samples)
    n_outliers = np.rint(outliers_fraction * n_samples)
    ground_truth = np.ones(n_samples, dtype=int)
    ground_truth[-n_outliers:] = 0

    np.random.seed(42)
    # Data generation
    offset = 2
    X1 = 0.3 * np.random.randn(0.5 * n_inliers, 2) - offset
    X2 = 0.3 * np.random.randn(0.5 * n_inliers, 2) + offset
    outliers1 = np.random.uniform(low=-6, high=-0, size=(0.5*n_outliers, 2))
    outliers2 = np.random.uniform(low=0, high=6, size=(0.5*n_outliers,2))
    X = np.r_[X1, X2, outliers1, outliers2]
    # Add outliers
    #X = np.r_[X, np.random.uniform(low=-6, high=6, size=(n_outliers, 2))]
    
    ocsvm.fit(X, np.repeat([1.0, -10.0], [n_inliers, n_outliers]))
    
    plt.figure()
    ax = plt.gca()
    plot_ocsvm_decision_boundary(ax, X, ground_truth, ocsvm)
    plt.show()
    
    
    ocsvm = svm.OneClassSVM(nu=0.95*(1-outliers_fraction)+0.05, kernel='rbf', gamma=0.9)
    ocsvm.fit(X, np.repeat([-10.0, 1.0], [n_inliers, n_outliers]))
    
    plt.figure()
    ax = plt.gca()
    plot_ocsvm_decision_boundary(ax, X, np.logical_not(ground_truth), ocsvm)
    plt.show()
    
    
    
    
    
    
    n_samples = 200
    outliers_fraction = 0.25

    ocsvm = svm.OneClassSVM(nu=0.95*outliers_fraction + 0.05, kernel="rbf", gamma=0.9)
   
    n_inliers = np.rint((1. - outliers_fraction) * n_samples)
    n_outliers = np.rint(outliers_fraction * n_samples)
    ground_truth = np.ones(n_samples, dtype=int)
    ground_truth[-n_outliers:] = 0

    np.random.seed(42)
    # Data generation
    offset = 2
    X1 = 0.3 * np.random.randn(0.5 * n_inliers, 2) - offset
    X2 = 0.3 * np.random.randn(0.5 * n_inliers, 2) + offset
    outliers1 = np.random.uniform(low=-3, high=-2, size=(0.5*n_outliers, 2))
    outliers2 = np.random.uniform(low=2, high=4, size=(0.5*n_outliers,2))
    X = np.r_[X1, X2, outliers1, outliers2]
    
    ocsvm = svm.OneClassSVM(nu=0.95*(1-outliers_fraction)+0.05, kernel='rbf', gamma=0.9)
    ocsvm.fit(X, np.repeat([-10.0, 1.0], [n_inliers, n_outliers]))
    
    plt.figure()
    ax = plt.gca()
    plot_ocsvm_decision_boundary(ax, X, np.logical_not(ground_truth), ocsvm)
    plt.show()
    