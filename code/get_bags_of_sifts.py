from PIL import Image
import numpy as np
from scipy.spatial import distance
import pickle
import scipy.spatial.distance as distance
from cyvlfeat.sift.dsift import dsift
from time import time
import pdb


from PIL import Image
import numpy as np
from scipy.spatial import distance
import pickle
from cyvlfeat.sift.dsift import dsift
from time import time

def get_bags_of_sifts(image_paths, vocab):
    # with open('vocab.pkl', 'rb') as handle:
    #     vocab = pickle.load(handle)

    image_feats = []

    start_time = time()
    print("Construct bags of sifts...")

    for path in image_paths:
        img = np.asarray(Image.open(path), dtype='float32')
        frames, descriptors = dsift(img, step=[5, 5], fast=False)  # Adjusted step and fast

        if descriptors is not None and len(descriptors)>0: #added check for when no descriptors are found. 
            dist = distance.cdist(vocab, descriptors, metric='euclidean')
            idx = np.argmin(dist, axis=0)
            hist, _ = np.histogram(idx, bins=len(vocab), range=(0,len(vocab))) #added range to be sure no out of range errors occur
            hist_norm = hist.astype(float)
            total_sum = sum(hist)
            if total_sum > 0:
                hist_norm /= total_sum
            else:
                hist_norm = np.zeros_like(hist)
        else:
            hist_norm = np.zeros(len(vocab)) #if no descriptors are found, return an empty histogram.

        image_feats.append(hist_norm)

    image_feats = np.asarray(image_feats)

    end_time = time()
    print("It takes ", (end_time - start_time), " to construct bags of sifts.")

    return image_feats




# def get_bags_of_sifts(image_paths):
#     ############################################################################
#     # TODO:                                                                    #
#     # This function assumes that 'vocab.pkl' exists and contains an N x 128    #
#     # matrix 'vocab' where each row is a kmeans centroid or visual word. This  #
#     # matrix is saved to disk rather than passed in a parameter to avoid       #
#     # recomputing the vocabulary every time at significant expense.            #
                                                                    
#     # image_feats is an N x d matrix, where d is the dimensionality of the     #
#     # feature representation. In this case, d will equal the number of clusters#
#     # or equivalently the number of entries in each image's histogram.         #
    
#     # You will want to construct SIFT features here in the same way you        #
#     # did in build_vocabulary.m (except for possibly changing the sampling     #
#     # rate) and then assign each local feature to its nearest cluster center   #
#     # and build a histogram indicating how many times each cluster was used.   #
#     # Don't forget to normalize the histogram, or else a larger image with more#
#     # SIFT features will look very different from a smaller version of the same#
#     # image.                                                                   #
#     ############################################################################
#     '''
#     Input : 
#         image_paths : a list(N) of training images
#     Output : 
#         image_feats : (N, d) feature, each row represent a feature of an image
#     '''
    
#     with open('vocab.pkl', 'rb') as handle:
#         vocab = pickle.load(handle)
    
#     image_feats = []
    
#     start_time = time()
#     print("Construct bags of sifts...")
    
#     for path in image_paths:
#         img = np.asarray(Image.open(path),dtype='float32')
#         frames, descriptors = dsift(img, step=[1,1], fast=True)
#         dist = distance.cdist(vocab, descriptors, metric='euclidean')
#         idx = np.argmin(dist, axis=0)
#         hist, bin_edges = np.histogram(idx, bins=len(vocab))
#         hist_norm = [float(i)/sum(hist) for i in hist]
        
#         image_feats.append(hist_norm)
        
#     image_feats = np.asarray(image_feats)
    
#     end_time = time()
#     print("It takes ", (start_time - end_time), " to construct bags of sifts.")
    
#     #############################################################################
#     #                                END OF YOUR CODE                           #
#     #############################################################################
    
#     return image_feats