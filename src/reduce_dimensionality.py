import numpy as np
from sklearn import decomposition
import matplotlib.pyplot as plt 
import matplotlib as mpl
import itertools
from sklearn.model_selection import KFold, train_test_split
from feature_extraction_eda import flatten_and_save_canny
import matplotlib.patches as mpatches

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
        data = np.load(f'{pose}.npy', allow_pickle=pickle)
        if not pickle:
            canny_data.append(flatten_and_save_canny(data))
        all_data.append(data)
    return all_data, canny_data

def scree_plot(pca_scree):
    '''
    Create Scree Plots in order to determine how many features to reduce images to using PCA
    '''
    plt.figure(1, figsize=(10, 6))
    plt.clf()
    plt.axes([.2, .2, .7, .7])
    plt.plot(pca_scree.explained_variance_, linewidth=2)
    plt.axis('tight')
    plt.xlabel('Number of Principal Components', fontsize=15)
    plt.ylabel('Explained Variance', fontsize=15)
    plt.title('Explained Variance For Number of Principal Components',
                fontsize=24)
    plt.savefig('../images/screevar.png', bbox='tight')
    total_variance = np.sum(pca_scree.explained_variance_)
    cum_variance = np.cumsum(pca_scree.explained_variance_)
    prop_var_expl = cum_variance/total_variance
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(prop_var_expl, linewidth=2, label='Explained variance')
    ax.axhline(0.9, label='90% goal', linestyle='--', color="black", linewidth=1)
    ax.set_ylabel('Cumulative Proportion of Explained Variance', fontsize=15)
    ax.set_xlabel('Number of Principal Components', fontsize=15)
    ax.set_title('Cumulative Explained Variance for Number of Principal Components',
                    fontsize=24)
    ax.legend()
    plt.savefig('../images/scree.png',bbox='tight')

def two_dim_pca(X, y):
    '''
    Plots 2D PCA factors and saves image
    '''
    fig, ax = plt.subplots(1, figsize=(12,8))
    ax.set_xlabel('Principal Component 1', fontsize = 18)
    ax.set_ylabel('Principal Component 2', fontsize = 18)
    ax.set_title('PCA 2 Components and Canny Filter', fontsize = 24)
    ax.scatter(X[:,0], X[:,1], c=y,cmap='bwr_r', alpha=0.6)
    ax.set_aspect('equal')
    red = mpatches.Patch(color='red', label='Mountain')
    blue = mpatches.Patch(color='blue', label='Downdog')
    plt.legend(handles=[red, blue], fontsize=15)
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

if __name__ == '__main__':
    #raw data combined
    raw_files = ['raw_downdog', 'raw_mountain', 'raw_file_downdog','raw_file_mountain']
    down = np.load('../data/cnn_data/downdog.npy',allow_pickle=False)
    mountain = np.load('../data/cnn_data/mountain.npy',allow_pickle=False)
    down_canny = flatten_and_save_canny(down)
    mount = flatten_and_save_canny(mountain)
    downdog_target = np.zeros((len(down_canny),1),dtype=int)
    mountain_target= np.ones((len(mount),1),dtype=int)

    #combine two poses into one dataset
    X = np.concatenate((down_canny, mount), axis=0)
    targets = np.concatenate((downdog_target, mountain_target), axis=0)
    y = np.ravel(targets)
    indeces = np.arange(len(X))

    # # #featurize into two components using PCA
    pca = decomposition.PCA(n_components=3)
    X_pca = pca.fit_transform(X)

    three_dim_pca(X_pca, y)
