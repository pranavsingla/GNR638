from datasets import load_dataset
from PIL import Image
import os
import random
from collections import defaultdict
import sys
# ---------------------------
# Parameters
# ---------------------------
output_dir = "data"  # Folder where the split folders will be created
random_seed = 42
random.seed(random_seed)

# ---------------------------
# Load UC Merced dataset from Hugging Face
# ---------------------------
print("Loading UC Merced dataset...")
ucmerced = load_dataset("blanchon/UC_Merced", split="train")
print("Total images:", len(ucmerced))


# ---------------------------
# Stratified Split per Class: 70% train, 10% val, 20% test
# ---------------------------
# Group samples by label
data_by_label = defaultdict(list)
for sample in ucmerced:
    # print(sample)
    data_by_label[sample["label"]].append(sample)
print(len(data_by_label))

train_samples = []
val_samples = []
test_samples = []

# For each class, split the samples.
for label, samples in data_by_label.items():
    n = len(samples)
    # For example, if n == 100, then:
    #   Train: 70, Val: 10, Test: 20
    n_train = 70
    n_val = 10
    # To ensure the test gets exactly the remainder, we compute:
    n_test = 20
    
    print(f"Class {label}: total {n}, train {n_train}, val {n_val}, test {n_test}")
    
    train_samples.extend(samples[:n_train])
    val_samples.extend(samples[n_train:n_train+n_val])
    test_samples.extend(samples[n_train+n_val:])


print("Final sizes:")
print("Train set size:", len(train_samples))
print("Validation set size:", len(val_samples))
print("Test set size:", len(test_samples))

# ---------------------------
# Save images to folders with structure:
# data/
#   train/<class>/image_XXXX.jpg
#   val/<class>/image_XXXX.jpg
#   test/<class>/image_XXXX.jpg
# ---------------------------
def save_split(samples, split_name):
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    # For each sample, save the image in the corresponding class folder
    # using the naming convention "image_XXXX.jpg".
    counters = {}  # To count images per class
    sanity = 0
    print(len(samples))
    for sample in samples:
        print("hi")
        label = sample["label"]
        # Create subfolder for this class if it doesn't exist.
        label_dir = os.path.join(split_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)
        # Update the counter for this label.
        count = counters.get(label, 0)
        file_name = "image_{:04d}.jpg".format(count)
        save_path = os.path.join(label_dir, file_name)
        # Ensure the image is a PIL Image.
        img = sample["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        img = img.convert("L") 
        try:
            print("saving")
            img.save(save_path, "JPEG")
            sanity+=1
        except Exception as e:
            print(f"Error saving {save_path}: {e}")
        counters[label] = count + 1
        
    return counters, sanity

print(save_split(train_samples, "train"))
save_split(val_samples, "val")
save_split(test_samples, "test")

print("Dataset saved in folder structure:")
print(os.path.abspath(output_dir))



# # from datasets import load_dataset
# # from PIL import Image
# # import os

# # ---------------------------
# # Load and split the dataset
# # ---------------------------
# print("Loading UC Merced dataset from Hugging Face...")
# ucmerced = load_dataset("blanchon/UC_Merced")
# # Assume the dataset is contained in the "train" split.
# full_dataset = ucmerced["train"]

# # Split off test set (20% overall)
# split_dataset = full_dataset.train_test_split(test_size=0.2, seed=42)
# train_val_set = split_dataset["train"]  # 80% of full
# test_set = split_dataset["test"]         # 20% of full

# # From the remaining 80%, split off validation set (10% overall).
# # 10%/80% = 0.125, so use test_size=0.125.
# split_train_val = train_val_set.train_test_split(test_size=0.125, seed=42)
# train_set = split_train_val["train"]  # ~70% overall
# val_set = split_train_val["test"]       # ~10% overall

# print("Train set size:", len(train_set))
# print("Validation set size:", len(val_set))
# print("Test set size:", len(test_set))

# # ---------------------------
# # Save images into folder structure
# # ---------------------------
# # Define the output folder for our new directory structure.
# output_dir = "ucm_data"  # You can change this to your desired path.
# splits = {
#     "train": train_set,
#     "val": val_set,
#     "test": test_set
# }

# # Iterate through each split.
# for split_name, dataset_split in splits.items():
#     split_dir = os.path.join(output_dir, split_name)
#     os.makedirs(split_dir, exist_ok=True)
#     print(f"Processing split '{split_name}'...")
    
#     # For each sample in the split, create a folder for its class if not already exists.
#     for idx, sample in enumerate(dataset_split):
#         # Assume the dataset contains "image" (a PIL Image) and "label" (a string).
#         # If the "image" field is not a PIL Image, convert it appropriately.
#         image = sample["image"]
#         if not isinstance(image, Image.Image):
#             image = Image.fromarray(image)
        
#         # Use the label to determine the subdirectory.
#         label = sample["label"]
#         print(label)
#         label_dir = os.path.join(split_dir, str(label))
#         os.makedirs(label_dir, exist_ok=True)
        
#         # Create a unique filename (e.g., using index and class name).
#         file_name = "image_{:04d}.jpg".format(idx)
#         save_path = os.path.join(label_dir, file_name)
        
#         # Save the image as a JPEG.
#         try:
#             image.save(save_path, "JPEG")
#         except Exception as e:
#             print(f"Error saving image {save_path}: {e}")

# print("Dataset saved in folder structure:")
# print(f"{output_dir}/")




# from datasets import load_dataset

# # Load the UC Merced dataset from Hugging Face
# ucmerced = load_dataset("blanchon/UC_Merced")
# # Assume the dataset is contained in the "train" split
# full_dataset = ucmerced["train"]

# # First, split off the test set (20% of the overall data)
# split_dataset = full_dataset.train_test_split(test_size=0.2, seed=42)
# train_val_set = split_dataset["train"]  # 80% of full
# test_set = split_dataset["test"]         # 20% of full

# # Now, from the remaining 80%, we want a validation set that is 10% of the full dataset.
# # Since 10/80 = 0.125, we split train_val_set with test_size=0.125 to get a validation set.
# split_train_val = train_val_set.train_test_split(test_size=0.125, seed=42)
# train_set = split_train_val["train"]  # ~70% of full
# val_set = split_train_val["test"]      # ~10% of full

# # Print sizes to verify the split proportions.
# print("Train set size:", len(train_set))
# print("Validation set size:", len(val_set))
# print("Test set size:", len(test_set))
