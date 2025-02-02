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

from sklearn.model_selection import KFold
from sklearn.manifold import TSNE
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
parser.add_argument('--feature', help='feature', type=str, default='dumy_feature')
parser.add_argument('--classifier', help='classifier', type=str, default='dumy_classifier')
args = parser.parse_args()

DATA_PATH = '../ucmerceddata/'

#This is the list of categories / directories to use. The categories are
#somewhat sorted by similarity so that the confusion matrix looks more
#structured (indoor and then urban and then rural).

# CATEGORIES = ['Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office',
#               'Industrial', 'Suburb', 'InsideCity', 'TallBuilding', 'Street',
#               'Highway', 'OpenCountry', 'Coast', 'Mountain', 'Forest']

CATEGORIES = ['agricultural',  'baseballdiamond',  'buildings', 'denseresidential', 'freeway', 'harbor', 'mediumresidential',
                'overpass',    'river',   'sparseresidential',  'tenniscourt', 'airplane', 'beach', 'chaparral',  
                'forest', 'golfcourse', 'intersection', 'mobilehomepark', 'parkinglot',  'runway',  'storagetanks']

CATE2ID = {v: k for k, v in enumerate(CATEGORIES)}

# ABBR_CATEGORIES = ['Kit', 'Sto', 'Bed', 'Liv', 'Off', 'Ind', 'Sub',
#                    'Cty', 'Bld', 'St', 'HW', 'OC', 'Cst', 'Mnt', 'For']
ABBR_CATEGORIES = ['agr', 'bas', 'bui', 'den', 'fre', 'har', 'med',
                   'ove', 'riv', 'spa', 'ten', 'air', 'bea', 'cha', 
                   'for', 'gol', 'int', 'mob', 'par', 'run', 'sto']


FEATURE = args.feature
# FEATUR  = 'bag of sift'

CLASSIFIER = args.classifier
# CLASSIFIER = 'support vector machine'

#number of training examples per category to use. Max is 100. For
#simplicity, we assume this is the number of test cases per category, as
#well.

NUM_TRAIN_PER_CAT = 1
NUM_TEST_PER_CAT = 1


def main():
    #This function returns arrays containing the file path for each train
    #and test image, as well as arrays with the label of each train and
    #test image. By default all four of these arrays will be 1500 where each
    #entry is a string.
    print("Getting paths and labels for all train and test data")
    train_image_paths, test_image_paths, train_image_labels, test_labels = \
        get_image_paths(DATA_PATH, CATEGORIES, NUM_TRAIN_PER_CAT, NUM_TEST_PER_CAT)

    # Set up K-fold cross-validation
    kf = KFold(n_splits=8, shuffle=True, random_state=42)

    # Store accuracy for different numbers of codewords
    val_accuracies = {}
    test_accuracies = {}


    for num_codewords in range(300, 601, 100):  # Range of codewords (300, ..., 600)
        val_fold_accuracies = []
        test_fold_accuracies = []
        for train_idx, val_idx in kf.split(train_image_paths):  # Splitting into train and validation
            # Select the training and validation paths
            train_paths = [train_image_paths[i] for i in train_idx]
            val_paths = [train_image_paths[i] for i in val_idx]
            train_labels = [train_image_labels[i] for i in train_idx]
            val_labels = [train_image_labels[i] for i in val_idx]

            # TODO Step 1:
            # Represent each image with the appropriate feature
            # Each function to construct features should return an N x d matrix, where
            # N is the number of paths passed to the function and d is the 
            # dimensionality of each image representation. See the starter code for
            # each function for more details.

            if FEATURE == 'tiny_image':
                # YOU CODE get_tiny_images.py 
                train_feats = get_tiny_images(train_paths)
                val_feats = get_tiny_images(val_paths)
                test_feats = get_tiny_images(test_image_paths)

            elif FEATURE == 'bag_of_sift':
                # YOU CODE build_vocabulary.py
                vocab_size = num_codewords   ### Vocab_size is up to you. Larger values will work better (to a point) but be slower to comput.
                vocab = build_vocabulary(train_paths, vocab_size)
                with open('vocab.pkl', 'wb') as handle:
                    pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

                # YOU CODE get_bags_of_sifts.py
                train_feats = get_bags_of_sifts(train_paths);
                with open('train_feats_1.pkl', 'wb') as handle:
                    pickle.dump(train_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
                val_feats  = get_bags_of_sifts(val_paths);
                with open('val_image_feats_1.pkl', 'wb') as handle:
                    pickle.dump(val_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
                test_feats  = get_bags_of_sifts(test_image_paths);
                with open('test_image_feats_1.pkl', 'wb') as handle:
                    pickle.dump(test_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)

            elif FEATURE == 'dumy_feature':
                train_feats = []
                val_feats = []
                test_feats = []

            else:
                raise NameError('Unknown feature type')

            # TODO Step 2: 
            # Classify each test image by training and using the appropriate classifier
            # Each function to classify test features will return an N x 1 array,
            # where N is the number of test cases and each entry is a string indicating
            # the predicted category for each test image. Each entry in
            # 'predicted_categories' must be one of the 15 strings in 'categories',
            # 'train_labels', and 'val_labels.

            if CLASSIFIER == 'nearest_neighbor':
                # YOU CODE nearest_neighbor_classify.py
                val_predicted_categories = nearest_neighbor_classify(train_feats, train_labels, val_feats)
                test_predicted_categories = nearest_neighbor_classify(train_feats, train_labels, test_feats)

            elif CLASSIFIER == 'support_vector_machine':
                # YOU CODE svm_classify.py
                val_predicted_categories = svm_classify(train_feats, train_labels, val_feats)
                test_predicted_categories = svm_classify(train_feats, train_labels, test_feats)

            elif CLASSIFIER == 'dumy_classifier':
                # The dummy classifier simply predicts a random category for
                # every test case
                val_predicted_categories = val_labels[:]
                shuffle(val_predicted_categories)
                test_predicted_categories = val_labels[:]
                shuffle(test_predicted_categories)
            else:
                raise NameError('Unknown classifier type')

            val_accuracy = float(len([x for x in zip(val_labels,val_predicted_categories) if x[0]== x[1]]))/float(len(val_labels))
            val_fold_accuracies.append(val_accuracy)
            test_accuracy = float(len([x for x in zip(test_labels,test_predicted_categories) if x[0]== x[1]]))/float(len(test_labels))
            test_fold_accuracies.append(test_accuracy)

         # Store the mean accuracy for the current number of codewords
        val_accuracies[num_codewords] = np.mean(val_fold_accuracies)
        test_accuracies[num_codewords] = np.mean(test_fold_accuracies)
        print("Num Codewords = ", num_codewords, "\nVal Accuracy = ", val_accuracies[num_codewords], "\nTest Accuracy = ", test_accuracies[num_codewords])
    
        for category in CATEGORIES:
            accuracy_each = float(len([x for x in zip(test_labels,test_predicted_categories) if x[0]==x[1] and x[0]==category]))/float(test_labels.count(category))
            print(str(category) + ': ' + str(accuracy_each))
        
        test_labels_ids = [CATE2ID[x] for x in test_labels]
        predicted_categories_ids = [CATE2ID[x] for x in test_predicted_categories]
        train_labels_ids = [CATE2ID[x] for x in train_labels]

        
        # Step 3: Build a confusion matrix and score the recognition system
        # You do not need to code anything in this section. 
    
        build_confusion_mtx(test_labels_ids, predicted_categories_ids, ABBR_CATEGORIES)
        visualize(CATEGORIES, test_image_paths, test_labels_ids, predicted_categories_ids, train_paths, train_labels_ids)
    
    # Plotting the accuracy vs. number of codewords
    plt.plot(val_accuracies.keys(), val_accuracies.values(), marker='o')
    plt.plot(test_accuracies.keys(), test_accuracies.values(), marker='x')
    plt.title('Accuracy vs Number of Codewords')
    plt.xlabel('Number of Codewords')
    plt.ylabel('Accuracy')
    plt.grid(True)

    # Highlight the optimal number of codewords
    optimal_num_codewords = max(val_accuracies, key=val_accuracies.get)
    optimal_accuracy = val_accuracies[optimal_num_codewords]
    plt.scatter(optimal_num_codewords, optimal_accuracy, color='red', label=f'Optimal: {optimal_num_codewords} codewords')

    plt.legend()
    plt.savefig('acc_codeword_plot.png')
    plt.close()

    keypoints_list, descriptors_list = extract_sift_keypoints(test_image_paths)
    plot_tsne(keypoints_list, 'sift_tsne_plot.png')


def extract_sift_keypoints(image_paths):
    sift = cv2.SIFT_create()
    keypoints_list = []
    descriptors_list = []
    
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        keypoints, descriptors = sift.detectAndCompute(img, None)
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)
    
    return keypoints_list, descriptors_list

def plot_tsne(keypoints_list, output_file):
    # Flatten the keypoints and perform t-SNE
    all_keypoints = np.vstack([keypt for keypt in keypoints_list if keypt is not None])
    tsne = TSNE(n_components=2, random_state=42)
    reduced_data = tsne.fit_transform(all_keypoints)

    # Plot t-SNE
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], s=1)
    plt.title('t-SNE visualization of SIFT keypoints')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.savefig(output_file)
    plt.close()

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
    #print('Normalized confusion matrix')
    #print(cm_normalized)
    plt.figure()
    plot_confusion_matrix(cm_normalized, abbr_categories, title='Normalized confusion matrix')

    plt.show()
     
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

if __name__ == '__main__':
    main()
