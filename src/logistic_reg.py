import numpy as np
from sklearn import decomposition
import matplotlib.pyplot as plt 
import matplotlib as mpl
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from feature_extraction_eda import flatten_and_save_canny
from sklearn.metrics import roc_curve, auc, confusion_matrix, roc_auc_score

plt.style.use('seaborn')
mpl.rcParams['figure.dpi'] = 100

def process_data(all_poses, pickle=False):
    '''
    Create canny filtered data, as vectorized images

    Returns both uncanny-ed and canny-ed data
    '''
    all_data = []
    canny_data = []
    for pose in all_poses:
        data = np.load(f'../data/{pose}.npy', allow_pickle=pickle)
        if not pickle:
            canny_data.append(flatten_and_save_canny(data))
        all_data.append(data)
    return all_data, canny_data

def two_dim_pca(X, y):
    '''
    Plots 2D PCA factors and saves image
    '''
    fig, ax = plt.subplots(1)
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('PCA with 2 Components', fontsize = 24)
    ax.scatter(X[:,0], X[:,1], c=y,cmap='bwr_r')
    plt.savefig('../images/PCA_plot_2.png')

def three_dim_pca(X, y):
    '''
    Plots 3D PCA factors and saves image
    '''
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
    '''
    Performs KFold Cross Validataion with a Logistic Regression Model
    
    Returns Train and Test Accuracy from cross validation
    '''
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

def we_will_roc_you(X_train, X_test, y_train, y_test):
    '''
    Plots and saves an ROC curve for logistic Regression and calculates the 
    total accuracy of the model as Area Under the Curve
    '''
    #train model
    model = LogisticRegression()
    model.fit(X_train,y_train)
    probabilities = model.predict_proba(X_test)[:,1]
    fpr, tpr, thresholds = roc_curve(y_test, probabilities)
    auc_score = round(roc_auc_score(y_test,probabilities), 4)
    #plot
    fig, ax = plt.subplots(1, figsize=(10,6))
    x = np.linspace(0,1, 100)
    ax.plot(fpr, tpr, label=f'AUC = {auc_score}')
    ax.plot(x, x, linestyle='--', color ='black', label='Random Guess')
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=16)
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=16)
    ax.set_title('ROC Curve with 3 Features', fontsize=24)
    plt.legend()
    plt.savefig('../images/roccurve_3.png',  bbox_inches='tight')
    return thresholds[fpr>0.2][0]

def con_matrix(y_hat, y_test, poses):
    '''
    Calculates and plots a confusion matrix for predicted and true labels
    '''
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

down = np.load('downdog.npy',allow_pickle=False)
mountain = np.load('mountain.npy',allow_pickle=False)
down_canny = flatten_and_save_canny(down)
mount = flatten_and_save_canny(mountain)
downdog_target = np.zeros((len(down_canny),1),dtype=int)
mountain_target= np.ones((len(mount),1),dtype=int)

#combine two poses into one dataset
X = np.concatenate((down_canny, mount), axis=0)
targets = np.concatenate((downdog_target, mountain_target), axis=0)
y = np.ravel(targets)
indeces = np.arange(len(X))

#featurize into three components using PCA
pca = decomposition.PCA(n_components=3)
X_pca = pca.fit_transform(X)

#featurize into two components using PCA
pca = decomposition.PCA(n_components=2)
X_pca2 = pca.fit_transform(X)

if __name__ == '__main__':
    #create train/test split with indeces for 2 features
    X_tr2, X_te2, y_tr2, y_te2, idx_tr2, idx_te2 = train_test_split(X_pca2,
                                                                    y,indeces,
                                                                    random_state=0)
    #create train/test split with indeces for 210 features    
    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(X_pca,
                                                                    y,indeces,
                                                                    random_state=0)
    #cross validation
    threshold = 0.53
    train_acc2, test_acc2 = crossVal(X_tr2, y_tr2, 5, 
                                        threshold=threshold)
    train_acc3, test_acc3 = crossVal(X_tr, y_tr, 5, 
                                        threshold=threshold)
    #ROC curves
    we_will_roc_you(X_tr2, X_te2, y_tr2, y_te2)
    we_will_roc_you(X_tr, X_te, y_tr, y_te)
    
    #logistic regression with 3 features:
    model = LogisticRegression()
    model.fit(X_tr2,y_tr2)
    probabilities = model.predict_proba(X_te2)[:,1]
    y_hat = (probabilities >= threshold).astype(int)
    print(model.coef_, model.intercept_)
    
    #overall accuracy
    total_acc = accuracy_score(y_te2, y_hat)
    print(total_acc)
    
    #confusion matrix
    poses = ['downdog', 'mountain']
    con_matrix(y_te, y_hat, poses)
