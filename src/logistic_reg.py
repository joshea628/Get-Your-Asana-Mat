import numpy as np
from sklearn import decomposition
import matplotlib.pyplot as plt 
import matplotlib as mpl
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from feature_extraction_eda import flatten_and_save_canny
from sklearn.metrics import roc_curve, auc, confusion_matrix

plt.style.use('seaborn')
mpl.rcParams['figure.dpi'] = 100

#create canny filter flattened data for each pose
downdog_data = np.load('../data/downdog.npy')
canny_downdog = flatten_and_save_canny(downdog_data, 'downdog')
downdog_target = np.zeros((len(canny_downdog),1),dtype=int)
mountain_data = np.load('../data/mountain.npy')
canny_mountain = flatten_and_save_canny(mountain_data, 'mountain')
mountain_target = np.ones((len(canny_mountain),1),dtype=int)

#combine two poses into one dataset
X = np.concatenate((canny_downdog, canny_mountain), axis=0)
targets = np.concatenate((downdog_target, mountain_target), axis=0)
y = np.ravel(targets)

#featurize into two components using PCA
pca = decomposition.PCA(n_components=2)
X_pca = pca.fit_transform(X)

def plot_pca(ax, X, y):
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('PCA with 2 Components', fontsize = 20)
    ax.scatter(X[:,0], X[:,1], c=colors,cmap='bwr_r')
    plt.savefig('../images/PCA_plot.png')

def crossVal(X, y, k, threshold=0.75):
    kf = KFold(n_splits=k)
    train_accuracy = []
    test_accuracy = []
   
    for train, test in kf.split(X):
        # Split into train and test
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        # Fit estimator
        model = LogisticRegression()
        model.fit(X_train, y_train)
        # Measure performance
        y_hat_trainprob = model.predict_proba(X_train)[:,1]
        y_hat_testprob = model.predict_proba(X_test)[:,1]
        y_hat_train = (y_hat_trainprob >= threshold).astype(int)
        y_hat_test = (y_hat_testprob >= threshold).astype(int)
        #metrics
        train_accuracy.append(accuracy_score(y_train, y_hat_train))
        test_accuracy.append(accuracy_score(y_test, y_hat_test))
    return np.mean(train_accuracy), np.mean(test_accuracy) 

def we_will_roc_you(X,y):
    model = LogisticRegression()
    
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    model.fit(X_train,y_train)
    probabilities = model.predict_proba(X_test)[:,1]

    fpr, tpr, thresholds = roc_curve(y_test, probabilities)
    x = np.linspace(0,1, 100)
    fig, ax = plt.subplots(1, figsize=(10,6))
    ax.plot(fpr, tpr)
    ax.plot(x, x, linestyle='--', color ='black', label='Random Guess')
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=16)
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=16)
    ax.set_title('ROC Curve', fontsize=18)
    plt.legend()
    plt.savefig('../images/roccurve.png',  bbox_inches='tight')

    return thresholds[fpr>0.3][0]

if __name__ == '__main__':
    # fig, ax = plt.subplots(1)
    # plot_pca(ax, X_pca, y)
    # plt.show()

    train_acc, test_acc = crossVal(X_pca, y, 5, threshold=0.75)
    print(train_acc, test_acc)

    we_will_roc_you(X_pca,y)
    plt.show()