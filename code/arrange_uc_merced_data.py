from sklearn.model_selection import train_test_split
from PIL import Image
import os

def prepare_uc_merced_data(input_dir, output_dir):
    classes = os.listdir(input_dir)
    for cls in classes:
        cls_dir = os.path.join(input_dir, cls)
        images = [img for img in os.listdir(cls_dir) if img.lower().endswith('.tif')]  # Process only .tif images

        train, test = train_test_split(images, test_size=0.2, random_state=42)

        for split, split_name in [(train, 'train'), (test, 'test')]:
            split_dir = os.path.join(output_dir, split_name, cls)
            os.makedirs(split_dir, exist_ok=True)

            for img in split:
                img_path = os.path.join(cls_dir, img)
                save_path = os.path.join(split_dir, img)  # Keep original filename
                
                # Load image and convert to grayscale
                with Image.open(img_path) as image:
                    gray_image = image.convert("L")  # Convert to grayscale (2D)
                    gray_image.save(save_path, "TIFF")  # Save as grayscale .tif

prepare_uc_merced_data('/home/spatel/dh602/Scene-Recognition-with-Bag-of-Words-1/UCMerced_LandUse/Images', 
                       '/home/spatel/dh602/Scene-Recognition-with-Bag-of-Words/ucmerceddata')


# from sklearn.model_selection import train_test_split
# from PIL import Image
# import os

# def prepare_uc_merced_data(input_dir, output_dir):
#     classes = os.listdir(input_dir)
#     for cls in classes:
#         cls_dir = os.path.join(input_dir, cls)
#         images = [img for img in os.listdir(cls_dir) if img.lower().endswith('.tif')]  # Process only .tif images

#         train, test = train_test_split(images, test_size=0.2, random_state=42)

#         for split, split_name in [(train, 'train'), (test, 'test')]:
#             split_dir = os.path.join(output_dir, split_name, cls)
#             os.makedirs(split_dir, exist_ok=True)

#             for img in split:
#                 img_path = os.path.join(cls_dir, img)
                
#                 # Load image and convert to grayscale
#                 with Image.open(img_path) as image:
#                     gray_image = image.convert("L")  # Convert to grayscale (2D)
                    
#                     # Save as .jpg with grayscale
#                     save_path = os.path.join(split_dir, img.replace(".tif", ".jpg"))
#                     gray_image.save(save_path, "JPEG")

# prepare_uc_merced_data('/home/spatel/dh602/Scene-Recognition-with-Bag-of-Words-1/UCMerced_LandUse/Images', 
#                        '/home/spatel/dh602/Scene-Recognition-with-Bag-of-Words/ucmerceddata')



# from sklearn.model_selection import train_test_split
# import os
# import shutil

# def prepare_uc_merced_data(input_dir, output_dir):
#     classes = os.listdir(input_dir)
#     for cls in classes:
#         cls_dir = os.path.join(input_dir, cls)
#         images = os.listdir(cls_dir)
#         train, test = train_test_split(images, test_size=0.2, random_state=42)

#         for split, split_name in [(train, 'train'), (test, 'test')]:
#             split_dir = os.path.join(output_dir, split_name, cls)
#             os.makedirs(split_dir, exist_ok=True)
#             for img in split:
#                 shutil.copy(os.path.join(cls_dir, img), os.path.join(split_dir, img))

# prepare_uc_merced_data('/home/spatel/dh602/Scene-Recognition-with-Bag-of-Words-1/UCMerced_LandUse/Images', '/home/spatel/dh602/Scene-Recognition-with-Bag-of-Words/ucmerceddata')
