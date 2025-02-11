import os
from glob import glob

def get_image_paths(data_path, categories, num_train_per_cat, num_test_per_cat, num_val_per_cat):
    num_categories = len(categories)

    train_image_paths = []
    test_image_paths = []
    val_image_paths = []

    train_labels = []
    test_labels = []
    val_labels = []

    for category in categories:
        # print(category)
        train_paths = glob(os.path.join(data_path, 'train', category, '*.jpg'))
        for i in range(num_train_per_cat):
            # print(i)
            # print(image_paths)
            train_image_paths.append(train_paths[i])
            train_labels.append(category)

        val_paths = glob(os.path.join(data_path, 'val', category, '*.jpg'))
        for i in range(num_val_per_cat):
            # print(i)
            # print(image_paths)
            val_image_paths.append(val_paths[i])
            val_labels.append(category)

        test_paths = glob(os.path.join(data_path, 'test', category, '*.jpg'))
        for i in range(num_test_per_cat):
            # print(i)
            # print(image_paths[i])
            test_image_paths.append(test_paths[i])
            test_labels.append(category)

    return train_image_paths, test_image_paths, val_image_paths, train_labels, val_labels, test_labels
