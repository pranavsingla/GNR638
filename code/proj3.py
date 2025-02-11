from __future__ import print_function
from random import shuffle
import os
import argparse
import pickle

from get_image_paths import get_image_paths
from get_tiny_images import get_tiny_images
from build_vocabulary import build_vocabulary
from get_bags_of_sifts import get_bags_of_sifts
from visualize import visualize

from nearest_neighbor_classify import nearest_neighbor_classify
from svm_classify import svm_classify
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Step 0: Set up parameters, category list, and image paths.

#For this project, you will need to report performance for three
#combinations of features / classifiers. It is suggested you code them in
#this order, as well:
# 1) Tiny image features and nearest neighbor classifier
# 2) Bag of sift features and nearest neighbor classifier
# 3) Bag of sift features and linear SVM classifier
#The starter code is initialized to 'placeholder' just so that the starter
#code does not crash when run unmodified and you can get a preview of how
#results are presented.

parser = argparse.ArgumentParser()
parser.add_argument('--feature', help='feature', type=str, default='bag_of_sift')
parser.add_argument('--classifier', help='classifier', type=str, default='nearest_neighbor')
args = parser.parse_args()

DATA_PATH = '../data/'

#This is the list of categories / directories to use. The categories are
#somewhat sorted by similarity so that the confusion matrix looks more
#structured (indoor and then urban and then rural).
CATEGORIES = [str(i) for i in range(21)]
print(CATEGORIES)
# CATEGORIES = ['Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office',
#               'Industrial', 'Suburb', 'InsideCity', 'TallBuilding', 'Street',
#               'Highway', 'OpenCountry', 'Coast', 'Mountain', 'Forest']

CATE2ID = {v: k for k, v in enumerate(CATEGORIES)}

ABBR_CATEGORIES = CATEGORIES


FEATURE = args.feature
# FEATURE  = 'bag of sift'

CLASSIFIER = args.classifier
# CLASSIFIER = 'support vector machine'

#number of training examples per category to use. Max is 100. For
#simplicity, we assume this is the number of test cases per category, as
#well.

NUM_TRAIN_PER_CAT = 70
NUM_VAL_PER_CAT = 10
NUM_TEST_PER_CAT = 20

def main():
    #This function returns arrays containing the file path for each train
    #and test image, as well as arrays with the label of each train and
    #test image. By default all four of these arrays will be 1500 where each
    #entry is a string.
    print("Getting paths and labels for all train, val and test data")
    train_image_paths, test_image_paths, val_image_paths, train_labels, val_labels, test_labels = get_image_paths(DATA_PATH, CATEGORIES, NUM_TRAIN_PER_CAT, NUM_TEST_PER_CAT, NUM_VAL_PER_CAT)

    # TODO Step 1:
    # Represent each image with the appropriate feature
    # Each function to construct features should return an N x d matrix, where
    # N is the number of paths passed to the function and d is the 
    # dimensionality of each image representation. See the starter code for
    # each function for more details.

    codebook_sizes = [100,200, 400, 800]
    val_accuracies = []
    for size in codebook_sizes:
        print(f"Evaluating with vocabulary size: {size}")
        vocab_temp = build_vocabulary(train_image_paths, size)
        train_feats_temp = get_bags_of_sifts(train_image_paths, vocab_temp)
        val_feats_temp = get_bags_of_sifts(val_image_paths, vocab_temp)
        if CLASSIFIER == 'nearest_neighbor':
            val_pred = nearest_neighbor_classify(train_feats_temp, train_labels, val_feats_temp)
        elif CLASSIFIER == 'support_vector_machine':
            val_pred = svm_classify(train_feats_temp, train_labels, val_feats_temp)
        acc = np.mean(np.array(val_pred) == np.array(val_labels))
        print(f"Val Acc: {acc}")
        val_accuracies.append(acc)
        
    #####PLotting
    plt.figure()
    plt.plot(codebook_sizes, val_accuracies, marker='o')
    plt.xlabel("Vocabulary Size (number of codewords)")
    plt.ylabel("Validation Accuracy")
    plt.title("Effect of Vocabulary Size on Validation Accuracy")
    plt.savefig(f'../results/{FEATURE}_{CLASSIFIER}_vocab_accuracy.png')
    val_accuracies = np.array(val_accuracies)  # Convert to NumPy array for efficiency
    num_codewords_list = np.array(codebook_sizes) # Ensure it's a NumPy array

    max_accuracy = np.max(val_accuracies)  # Find the maximum accuracy
    max_indices = np.where(val_accuracies == max_accuracy)[0]  # Indices where accuracy equals max_accuracy
    num_max_codewords = num_codewords_list[max_indices][0]  

    print(f"Going ahead with {num_max_codewords} codewords")

    vocab_max = build_vocabulary(train_image_paths, int(num_max_codewords))
    test_feats = get_bags_of_sifts(test_image_paths, vocab_max)
    train_feats_max = get_bags_of_sifts(train_image_paths, vocab_max)

    if CLASSIFIER == 'nearest_neighbor':
        test_pred = nearest_neighbor_classify(train_feats_max, train_labels, test_feats)
    elif CLASSIFIER == 'support_vector_machine':
        test_pred = svm_classify(train_feats_max, train_labels, test_feats)
    print("PREDICTED:", test_pred)
    acc = np.mean(np.array(test_pred) == np.array(test_labels))
    print("Accuracy: ", acc)
    #Metrics

    for category in CATEGORIES:
        accuracy_each = float(len([x for x in zip(test_labels,test_pred) if x[0]==x[1] and x[0]==category]))/float(test_labels.count(category))
        print(str(category) + ': ' + str(accuracy_each))
    
    test_labels_ids = [CATE2ID[x] for x in test_labels]
    predicted_categories_ids = [CATE2ID[x] for x in test_pred]
    
    # Step 3: Build a confusion matrix and score the recognition system
    # You do not need to code anything in this section. 
    print("plotting")
    build_confusion_mtx(test_labels_ids, predicted_categories_ids, ABBR_CATEGORIES)
    tsne_plot(train_image_paths, vocab=vocab_max)

    # vocab_size = 400 
    # if FEATURE == 'tiny_image':
    #     # YOU CODE get_tiny_images.py 
    #     train_image_feats = get_tiny_images(train_image_paths)
    #     test_image_feats = get_tiny_images(test_image_paths)

    # elif FEATURE == 'bag_of_sift':
    #     # YOU CODE build_vocabulary.py
    #     if os.path.isfile('vocab.pkl') is False:
    #         print('No existing visual word vocabulary found. Computing one from training images\n')
    #           ### Vocab_size is up to you. Larger values will work better (to a point) but be slower to comput.
    #         vocab = build_vocabulary(train_image_paths, vocab_size)
            
    #         with open('vocab.pkl', 'wb') as handle:
    #             pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #     else:
    #         with open('vocab.pkl', 'rb') as handle:
    #             vocab = pickle.load(handle)

    #     if os.path.isfile('train_image_feats_1.pkl') is False:
    #         # YOU CODE get_bags_of_sifts.py
    #         train_image_feats = get_bags_of_sifts(train_image_paths);
    #         with open('train_image_feats_1.pkl', 'wb') as handle:
    #             pickle.dump(train_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #     else:
    #         with open('train_image_feats_1.pkl', 'rb') as handle:
    #             train_image_feats = pickle.load(handle)

    #     if os.path.isfile('test_image_feats_1.pkl') is False:
    #         test_image_feats  = get_bags_of_sifts(test_image_paths);
    #         with open('test_image_feats_1.pkl', 'wb') as handle:
    #             pickle.dump(test_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #     else:
    #         with open('test_image_feats_1.pkl', 'rb') as handle:
    #             test_image_feats = pickle.load(handle)
    # elif FEATURE == 'dumy_feature':
    #     train_image_feats = []
    #     test_image_feats = []
    # else:
    #     raise NameError('Unknown feature type')

    # # TODO Step 2: 
    # # Classify each test image by training and using the appropriate classifier
    # # Each function to classify test features will return an N x 1 array,
    # # where N is the number of test cases and each entry is a string indicating
    # # the predicted category for each test image. Each entry in
    # # 'predicted_categories' must be one of the 15 strings in 'categories',
    # # 'train_labels', and 'test_labels.

    # if CLASSIFIER == 'nearest_neighbor':
    #     # YOU CODE nearest_neighbor_classify.py
    #     predicted_categories = nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats)

    # elif CLASSIFIER == 'support_vector_machine':
    #     # YOU CODE svm_classify.py
    #     predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats)

    # elif CLASSIFIER == 'dumy_classifier':
    #     # The dummy classifier simply predicts a random category for
    #     # every test case
    #     predicted_categories = test_labels[:]
    #     shuffle(predicted_categories)
    # else:
    #     raise NameError('Unknown classifier type')

    # accuracy = float(len([x for x in zip(test_labels,predicted_categories) if x[0]== x[1]]))/float(len(test_labels))
    # print("Accuracy = ", accuracy)
    
    # for category in CATEGORIES:
    #     accuracy_each = float(len([x for x in zip(test_labels,predicted_categories) if x[0]==x[1] and x[0]==category]))/float(test_labels.count(category))
    #     print(str(category) + ': ' + str(accuracy_each))
    
    # test_labels_ids = [CATE2ID[x] for x in test_labels]
    # predicted_categories_ids = [CATE2ID[x] for x in predicted_categories]
    # train_labels_ids = [CATE2ID[x] for x in train_labels]
    
    # # Step 3: Build a confusion matrix and score the recognition system
    # # You do not need to code anything in this section. 
    # print("plotting")
    # build_confusion_mtx(test_labels_ids, predicted_categories_ids, ABBR_CATEGORIES)
    # tsne_plot(train_image_paths, vocab=vocab)
    






def build_confusion_mtx(test_labels_ids, predicted_categories, abbr_categories):
    # Compute confusion matrix
    cm = confusion_matrix(test_labels_ids, predicted_categories)
    np.set_printoptions(precision=2)
    '''
    print('Confusion matrix, without normalization')
    print(cm)
    plt.figure()
    plot_confusion_matrix(cm, CATEGORIES)
    '''
    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # print('Normalized confusion matrix')
    # print(cm_normalized)
    plt.figure()
    plot_confusion_matrix(cm_normalized, abbr_categories, title='Normalized confusion matrix')
    plt.savefig(f'../results/{FEATURE}_{CLASSIFIER}_conf_mtx')
     
def plot_confusion_matrix(cm, category, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(category))
    plt.xticks(tick_marks, category, rotation=45)
    plt.yticks(tick_marks, category)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from PIL import Image
from cyvlfeat.sift.dsift import dsift
from scipy.spatial.distance import cdist

def tsne_plot(image_paths, num_samples=10000, step=[5,5], vocab=None):
    """
    Extracts SIFT descriptors from a list of image paths, optionally assigns cluster labels
    using a provided vocabulary (from a k-means clustering step), applies t-SNE to reduce the 
    descriptors to 2D, and produces a colored scatter plot.
    
    Parameters:
    - image_paths: list of image file paths.
    - num_samples: maximum number of descriptors to visualize.
    - step: step size used in dsift (adjust for speed/accuracy tradeoff).
    - vocab: (optional) vocabulary matrix (each row is a codeword). If provided, each SIFT 
             descriptor is assigned a cluster label based on its nearest codeword.
             
    The output plot shows points colored by their cluster assignment if vocab is provided.
    """
    descriptors_list = []
    cluster_labels_list = []  # To hold the cluster labels for each descriptor

    for path in image_paths:
        try:
            # Open image in grayscale
            img = Image.open(path).convert('L')
            img_np = np.array(img, dtype='float32')
            # Extract SIFT descriptors using dsift (ensure cyvlfeat is properly installed)
            _, descriptors = dsift(img_np, step=step, fast=True)
            if descriptors is not None:
                descriptors_list.append(descriptors)
                if vocab is not None:
                    # For each descriptor, compute its distance to each codeword
                    dists = cdist(vocab, descriptors, metric='euclidean')
                    # Nearest codeword index becomes the cluster label
                    labels = np.argmin(dists, axis=0)
                    cluster_labels_list.append(labels)
        except Exception as e:
            print("Error processing", path, ":", e)
    
    # Combine all descriptors from all images into one array.
    if len(descriptors_list) == 0:
        print("No descriptors extracted.")
        return
    descriptors_all = np.concatenate(descriptors_list, axis=0)
    
    if vocab is not None:
        cluster_labels_all = np.concatenate(cluster_labels_list, axis=0)
    else:
        cluster_labels_all = None

    # Randomly sample a subset of descriptors to visualize
    total_descriptors = descriptors_all.shape[0]
    if total_descriptors > num_samples:
        indices = np.random.choice(total_descriptors, num_samples, replace=False)
        descriptors_sampled = descriptors_all[indices]
        if cluster_labels_all is not None:
            cluster_labels_sampled = cluster_labels_all[indices]
        else:
            cluster_labels_sampled = None
    else:
        descriptors_sampled = descriptors_all
        cluster_labels_sampled = cluster_labels_all

    print("Running t-SNE on {} descriptors...".format(descriptors_sampled.shape[0]))
    # Run t-SNE with chosen hyperparameters. You can adjust perplexity or learning_rate if needed.
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    descriptors_2d = tsne.fit_transform(descriptors_sampled)
    
    plt.figure(figsize=(10, 8))
    if cluster_labels_sampled is not None:
        # Determine the number of unique clusters and set a discrete colormap.
        unique_labels = np.unique(cluster_labels_sampled)
        num_clusters = len(unique_labels)
        cmap = plt.get_cmap('tab20', num_clusters)
        scatter = plt.scatter(descriptors_2d[:, 0], descriptors_2d[:, 1],
                              c=cluster_labels_sampled, cmap=cmap, s=5, alpha=0.7)
        cbar = plt.colorbar(scatter, ticks=range(num_clusters))
        cbar.set_label('Cluster Label')
        plt.title("t-SNE Visualization of SIFT Descriptors\n(Colored by Cluster Assignment)")
    else:
        plt.scatter(descriptors_2d[:, 0], descriptors_2d[:, 1], s=5, alpha=0.7)
        plt.title("t-SNE Visualization of SIFT Descriptors")
    
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.tight_layout()
    # Save the figure using global variables FEATURE and CLASSIFIER, which should be defined in your main code.
    plt.savefig(f'../results/{FEATURE}_{CLASSIFIER}_tsne.png')
    plt.show()







# def tsne_plot(image_paths,vocab_size, num_samples=10000, step=[5,5]):
#     """
#     Extracts SIFT descriptors from a list of image paths, randomly samples
#     a subset of descriptors, and applies t-SNE to visualize them in 2D.
    
#     Parameters:
#     - image_paths: list of image file paths.
#     - num_samples: maximum number of descriptors to visualize.
#     - step: step size used in dsift (adjust for speed/accuracy tradeoff).
#     """
#     descriptors_list = []
    
#     for path in image_paths:
#         try:
#             img = Image.open(path).convert('L')
#             img_np = np.array(img, dtype='float32')
#             # Extract SIFT descriptors with the given step size
#             _, descriptors = dsift(img_np, step=step, fast=True)
#             if descriptors is not None:
#                 descriptors_list.append(descriptors)
#         except Exception as e:
#             print("Error processing", path, ":", e)
            
#     # Combine descriptors from all images
#     descriptors_all = np.concatenate(descriptors_list, axis=0)
    
#     # Sample a subset for t-SNE if needed
#     if descriptors_all.shape[0] > num_samples:
#         indices = np.random.choice(descriptors_all.shape[0], num_samples, replace=False)
#         descriptors_sampled = descriptors_all[indices]
#     else:
#         descriptors_sampled = descriptors_all
        
#     print("Running t-SNE on {} descriptors...".format(descriptors_sampled.shape[0]))
#     tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
#     descriptors_2d = tsne.fit_transform(descriptors_sampled)
    
#     plt.figure(figsize=(8, 6))
#     plt.scatter(descriptors_2d[:, 0], descriptors_2d[:, 1], s=1, alpha=0.5)
#     plt.title("t-SNE Visualization of SIFT Descriptors")
#     plt.xlabel("Dimension 1")
#     plt.ylabel("Dimension 2")
#     plt.savefig(f'../results/{FEATURE}_{CLASSIFIER}_{vocab_size}_tsne')
#     plt.show()

# Example usage:
# tsne_plot(train_image_paths)  # You can pass train_image_paths or a subset of them.


if __name__ == '__main__':
    main()
