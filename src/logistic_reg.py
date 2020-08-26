import numpy as np
from sklearn import decomposition
import matplotlib.pyplot as plt 
import matplotlib as mpl
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from feature_extraction_eda import flatten_and_save_canny
from sklearn.metrics import roc_curve, auc, confusion_matrix

plt.style.use('seaborn')
mpl.rcParams['figure.dpi'] = 100

def two_dim_pca(X, y):
    fig, ax = plt.subplots(1)
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('PCA with 2 Components', fontsize = 24)
    ax.scatter(X[:,0], X[:,1], c=y,cmap='bwr_r')
    plt.savefig('../images/PCA_plot_2.png')

def three_dim_pca(X, y):
    fig = plt.figure(figsize=(10,12))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_zlabel('Principal Component 3', fontsize = 15)
    ax.set_title('PCA with 3 Components', fontsize = 24)
    ax.scatter(X[:,0], X[:,1], X[:,2], c=y,cmap='bwr_r')
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    for line in ax.xaxis.get_ticklines():
        line.set_visible(False)
    for line in ax.yaxis.get_ticklines():
        line.set_visible(False)
    for line in ax.zaxis.get_ticklines():
        line.set_visible(False)
    plt.savefig('../images/PCA_plot_3.png')

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
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    #train model
    model = LogisticRegression()
    model.fit(X_train,y_train)
    probabilities = model.predict_proba(X_test)[:,1]
    fpr, tpr, thresholds = roc_curve(y_test, probabilities)
    #plot
    fig, ax = plt.subplots(1, figsize=(10,6))
    x = np.linspace(0,1, 100)
    ax.plot(fpr, tpr)
    ax.plot(x, x, linestyle='--', color ='black', label='Random Guess')
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=16)
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=16)
    ax.set_title('ROC Curve with 3 Features', fontsize=24)
    plt.legend()
    plt.savefig('../images/roccurve_3.png',  bbox_inches='tight')
    return thresholds[fpr>0.2][0]

def con_matrix(y_hat, y_test, poses):
    cm = confusion_matrix(y_test, y_hat)
    fig, ax = plt.subplots(1)
    p = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title('Confusion Matrix with 3 Features',fontsize=24)
    plt.colorbar(p)
    tick_marks = np.arange(len(poses))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(poses, rotation=0)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(poses)
    ax.grid(False)
    cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis],2)
    thresh = cm.max() / 1.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, cm[i, j],
                 horizontalalignment="center", size = 24,
                 color="white" if cm[i, j] > thresh else "black")
    
    ax.set_ylabel('True label',fontsize=24)
    ax.set_xlabel('Predicted label',fontsize=24)
    plt.tight_layout()
    plt.savefig('../images/confusion_matrix_3.png',  bbox_inches='tight')
    return cm

if __name__ == '__main__':
    #create canny filter flattened data for each pose
    downdog_data = np.load('../data/downdog.npy')
    mountain_data = np.load('../data/mountain.npy')
    canny_downdog = flatten_and_save_canny(downdog_data)
    canny_mountain = flatten_and_save_canny(mountain_data)
    downdog_target = np.zeros((len(canny_downdog),1),dtype=int)
    mountain_target = np.ones((len(canny_mountain),1),dtype=int)

    #combine two poses into one dataset
    X = np.concatenate((canny_downdog, canny_mountain), axis=0)
    targets = np.concatenate((downdog_target, mountain_target), axis=0)
    y = np.ravel(targets)

    #featurize into two components using PCA
    pca = decomposition.PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    two_dim_pca(X_pca, y)

    #featurize into 3 components using PCA
    pca = decomposition.PCA(n_components=3)
    X_pca3 = pca.fit_transform(X)
    three_dim_pca(X_pca3, y)

    #create holdout set - train/test split
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X_pca,y)
    X_train3, X_test3, y_train3, y_test3 = train_test_split(X_pca3,y)

    #cross validation
    threshold = 0.53
    train_acc2, test_acc2 = crossVal(X_train2, y_train2, 5, threshold=threshold)
    train_acc3, test_acc3 = crossVal(X_train3, y_train3, 5, threshold=threshold)
    print(train_acc2, test_acc2)
    print(train_acc3, test_acc3)

    #ROC curves
    we_will_roc_you(X_train2,y_train2)
    thresh = we_will_roc_you(X_train3,y_train3)
    print(thresh)

    #logistic regression with 3 features:
    model = LogisticRegression()
    model.fit(X_train3,y_train3)
    probabilities = model.predict_proba(X_test3)[:,1]
    y_hat = (probabilities >= threshold).astype(int)
    print(y_hat)
    print(y_test3)

    #confusion matrix
    poses = ['downdog', 'mountain']
    con_matrix(y_test3, y_hat, poses)
    plt.show()
    