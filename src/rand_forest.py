import numpy as np
import matplotlib.pyplot as plt 
import matplotlib as mpl
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, roc_auc_score
from logistic_reg import X_pca3, indeces, y, X_raw
import itertools

plt.style.use('seaborn')
mpl.rcParams['figure.dpi'] = 100

def grid_search(X_train, X_test, y_train, y_test):
    '''
    Randomized grid search for best RF parameters
    '''
    rf = RandomForestClassifier(n_estimators=100,
                                n_jobs=-1,
                                random_state=1)
    rf.fit(X_train, y_train)
    rf_ys = rf.predict(X_test)
    random_forest_grid = {'max_depth': [5, 10],
                        'max_features': ['sqrt', 'log2', None],
                        'min_samples_split': [2, 4, 5],
                        'min_samples_leaf': [1, 5],
                        'bootstrap': [True, False],
                        'n_estimators': [5, 10, 20],
                        'random_state': [1]}
    rf_gridsearch = RandomizedSearchCV(RandomForestClassifier(),
                                random_forest_grid,
                                n_iter = 200,
                                n_jobs=-1,
                                verbose=True,
                                scoring='accuracy')
    rf_gridsearch.fit(X_train, y_train)
    print("Random Forest best parameters:", rf_gridsearch.best_params_)
    best_rf_model = rf_gridsearch.best_estimator_
    return best_rf_model

def crossVal(model, X, y, k, threshold=0.75):
    '''
    Cross validation for Random Forest Model
    
    Returns train and test accuracy from cross validation
    '''
    kf = KFold(n_splits=k)
    train_accuracy = []
    test_accuracy = []  
    for train, test in kf.split(X):
        # Split into train and test
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        # Fit estimator
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

def we_will_roc_you(model, X_train, y_train, X_test, y_test):
    '''
    Creates ROC curve for RF model, calculates the Area Under the Curve
    '''
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
    ax.set_title('ROC Curve for Random Forest', fontsize=24)
    plt.legend()
    plt.savefig('../images/roccurve_random_forest.png',  bbox_inches='tight')
    return thresholds[fpr>0.25][0]

def con_matrix(y_hat, y_test, poses):
    '''
    Calculates and plots a confusion matrix for predicted and true labels
    '''
    cm = confusion_matrix(y_test, y_hat)
    fig, ax = plt.subplots(1)
    p = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title('Confusion Matrix for Random Forest',fontsize=24)
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
    plt.savefig('../images/confusion_matrix_RF.png',  bbox_inches='tight')
    return cm

if __name__ == '__main__':
    #create train/test split with indeces for 3 features    
    X_tr3, X_te3, y_tr3, y_te3, idx_tr3, idx_te3 = train_test_split(X_pca3,
                                                                    y,indeces,
                                                                    random_state=0)
    #perform grid search for best RF parameters
    model_pca3 = grid_search(X_tr3, X_te3, y_tr3, y_te3)
    #cross validation with RF model
    threshold=0.4
    train_acc, test_acc = crossVal(model_pca3, X_pca3, y, 5, threshold=threshold)
    print(train_acc, test_acc)
    #fit RF model 
    model_pca3.fit(X_tr3, y_tr3)
    probabilities = model_pca3.predict_proba(X_te3)[:,1]
    y_hat = (probabilities >= threshold).astype(int)
    print(y_hat)
    print(y_te3)
    total_acc = accuracy_score(y_te3, y_hat)
    print(total_acc)
    #ROC Curve
    #we_will_roc_you(model_pca3, X_tr3, y_tr3, X_te3, y_te3)
    #confusion matrix
    poses = ['downdog', 'mountain']
    #con_matrix(y_hat, y_te3, poses)
    #plt.show()
    print(len(y_te3))
    #show incorrect positives
    fig, ax = plt.subplots(1)
    display = X_raw[461]
    ax.imshow(display)
    ax.set_axis_off()
    plt.savefig('../images/actual_mountain_6.png')
    plt.show()  
    #model labeled mountain, actual downdog
    #indeces [119,110,105]
    print(idx_te3[118],idx_te3[109],idx_te3[104]) # 155, 171, 65
    #model labeled downdog, actual mountain
    #indeces [108, 107,104]
    print(idx_te3[107],idx_te3[106],idx_te3[103]) # 445, 261, 461