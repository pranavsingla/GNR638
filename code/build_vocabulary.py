from PIL import Image
import numpy as np
from cyvlfeat.sift.dsift import dsift
from cyvlfeat.kmeans import kmeans
from time import time

import pdb

#This function will sample SIFT descriptors from the training images,
#cluster them with kmeans, and then return the cluster centers.

from PIL import Image
import numpy as np
from cyvlfeat.sift.dsift import dsift
from cyvlfeat.kmeans import kmeans
from time import time

def build_vocabulary(image_paths, vocab_size):
    bag_of_features = []

    print("Extract SIFT features")

    for path in image_paths:
        try:  # Try to open and process the image
            img = np.asarray(Image.open(path), dtype='float32')
            frames, descriptors = dsift(img, step=[5, 5], fast=True)

            if descriptors is not None and len(descriptors) > 0:
                bag_of_features.append(descriptors)
            else:
                print(f"Warning: No SIFT features found in {path}")  # Informative message

        except Exception as e:  # Catch any exceptions during image processing
            print(f"Error processing image {path}: {e}")  # Print the error
            continue  # Skip to the next image

    if bag_of_features:  # Check if there are any descriptors at all
        bag_of_features = np.concatenate(bag_of_features, axis=0).astype('float32')

        # Random sampling (if needed)
        if bag_of_features.size > 0: #check if bag_of_features is not empty after the concatenation and filtering.
            sample_size = min(100000, bag_of_features.shape[0])  # Adjust sample size
            random_indices = np.random.choice(bag_of_features.shape[0], size=sample_size, replace=False)
            bag_of_features = bag_of_features[random_indices]

            print("Compute vocab")
            start_time = time()
            vocab = kmeans(bag_of_features, vocab_size, initialization="PLUSPLUS")
            end_time = time()
            print("It takes ", (end_time - start_time), " to compute vocab.")
        else:
            print("Warning: No SIFT features found in any image.")
            vocab = np.array([])

    else:
        print("Warning: No SIFT features found in any image.")
        vocab = np.array([])

    return vocab














# def build_vocabulary(image_paths, vocab_size):
#     ##################################################################################
#     # TODO:                                                                          #
#     # Load images from the training set. To save computation time, you don't         #
#     # necessarily need to sample from all images, although it would be better        #
#     # to do so. You can randomly sample the descriptors from each image to save      #
#     # memory and speed up the clustering. Or you can simply call vl_dsift with       #
#     # a large step size here.                                                        #
#     #                                                                                #
#     # For each loaded image, get some SIFT features. You don't have to get as        #
#     # many SIFT features as you will in get_bags_of_sift.py, because you're only     #
#     # trying to get a representative sample here.                                    #
#     #                                                                                #
#     # Once you have tens of thousands of SIFT features from many training            #
#     # images, cluster them with kmeans. The resulting centroids are now your         #
#     # visual word vocabulary.                                                        #
#     ##################################################################################
#     ##################################################################################
#     # NOTE: Some useful functions                                                    #
#     # This function will sample SIFT descriptors from the training images,           #
#     # cluster them with kmeans, and then return the cluster centers.                 #
#     #                                                                                #
#     # Function : dsift()                                                             #
#     # SIFT_features is a N x 128 matrix of SIFT features                             #
#     # There are step, bin size, and smoothing parameters you can                     #
#     # manipulate for dsift(). We recommend debugging with the 'fast'                 #
#     # parameter. This approximate version of SIFT is about 20 times faster to        #
#     # compute. Also, be sure not to use the default value of step size. It will      #
#     # be very slow and you'll see relatively little performance gain from            #
#     # extremely dense sampling. You are welcome to use your own SIFT feature.        #
#     #                                                                                #
#     # Function : kmeans(X, K)                                                        #
#     # X is a M x d matrix of sampled SIFT features, where M is the number of         #
#     # features sampled. M should be pretty large!                                    #
#     # K is the number of clusters desired (vocab_size)                               #
#     # centers is a d x K matrix of cluster centroids.                                #
#     #                                                                                #
#     # NOTE:                                                                          #
#     #   e.g. 1. dsift(img, step=[?,?], fast=True)                                    #
#     #        2. kmeans( ? , vocab_size)                                              #  
#     #                                                                                #
#     # ################################################################################
#     '''
#     Input : 
#         image_paths : a list of training image path
#         vocal size : number of clusters desired
#     Output :
#         Clusters centers of Kmeans
#     '''

#     '''   
#     bag_of_features = []
#     print("Extract SIFT features")
#     pdb.set_trace()
#     for path in image_paths:
#         img = np.asarray(Image.open(path),dtype='float32')
#         frames, descriptors = dsift(img, step=[5,5], fast=True)
#         bag_of_features.append(descriptors)
#     bag_of_features = np.concatenate(bag_of_features, axis=0).astype('float32')
#     pdb.set_trace()
#     print("Compute vocab")
#     start_time = time()
#     vocab = kmeans(bag_of_features, vocab_size, initialization="PLUSPLUS")        
#     end_time = time()
#     print("It takes ", (start_time - end_time), " to compute vocab.")
#     '''
#     bag_of_features = []
    
#     print("Extract SIFT features")
    
#     #The Python Debugger
#     #pdb.set_trace()
    
#     for path in image_paths:
#         img = np.asarray(Image.open(path),dtype='float32')
#         frames, descriptors = dsift(img, step=[8,8], fast=True)
#         bag_of_features.append(descriptors)
#     bag_of_features = np.concatenate(bag_of_features, axis=0).astype('float32')
#     #pdb.set_trace()
    
#     print("Compute vocab")
#     start_time = time()
#     vocab = kmeans(bag_of_features, vocab_size, initialization="PLUSPLUS")        
#     end_time = time()
#     print("It takes ", (start_time - end_time), " to compute vocab.")
    
    
#     ##################################################################################
#     #                                END OF YOUR CODE                                #
#     ##################################################################################
    
#     return vocab