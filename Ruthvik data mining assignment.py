#!/usr/bin/env python
# coding: utf-8

# In[27]:


import os
import glob
from PIL import Image
from pathlib import Path
import xml.etree.ElementTree as ET

dog_images_dir = glob.glob(r"C:/Users/ruthv/DataMining/images/*/*")
annotations_dir = glob.glob(r'C:/Users/ruthv/DataMining/Annotations/*/*')
cropped_images_dir = r'C:/Users/ruthv/DataMining/cropped'

def get_bounding_boxes(annot):
    tree = ET.parse(annot)
    root = tree.getroot()
    objects = root.findall('object')
    bbox = []
    for o in objects:
        bndbox = o.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        bbox.append((xmin, ymin, xmax, ymax))
    return bbox

def get_image(annot):
    img_path = r'C:/Users/ruthv/DataMining/images'
    file = os.path.normpath(annot).split(os.path.sep)
    img_filename = os.path.join(img_path, file[-2], file[-1].replace('.xml', '.jpg'))
    return img_filename

print(len(dog_images_dir))
for i in range(len(dog_images_dir)):
    try:
        bbox = get_bounding_boxes(annotations_dir[i])
        dog = get_image(annotations_dir[i])
        im = Image.open(dog)
        print(im)
        for j in range(len(bbox)):
            im2 = im.crop(bbox[j])
            im2 = im2.resize((128, 128), Image.ANTIALIAS)
            new_path = dog.replace('images', 'cropped')
            new_path, _ = os.path.splitext(new_path)
            new_path = new_path + '-' + str(j) + '.jpg'
            head, tail = os.path.split(new_path)
            Path(head).mkdir(parents=True, exist_ok=True)
            im2.save(new_path)
    except Exception as e:
        print(f"Error processing {annotations_dir[i]}: {e}")


# In[28]:


import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import color

# Function to get the list of images for each class
def get_images_per_class(images_dir):
    images_per_class = {}
    for image_path in images_dir:
        class_name = os.path.basename(os.path.dirname(image_path))
        if class_name not in images_per_class:
            images_per_class[class_name] = []
        images_per_class[class_name].append(image_path)
    return images_per_class

# Function to choose two images from each class
def choose_two_images_per_class(images_per_class):
    selected_images = {}
    for class_name, images in images_per_class.items():
        selected_images[class_name] = images[:2]
    return selected_images

# Function to convert color images to grayscale
def convert_to_grayscale(images):
    grayscale_images = []
    for image_path in images:
        image = Image.open(image_path)
        grayscale_image = color.rgb2gray(np.array(image))
        grayscale_images.append(grayscale_image)
    return grayscale_images

# Path to the directory containing images
dog_images_dir = glob.glob(r"C:/Users/ruthv/DataMining/images/*/*")

# Get list of images per class
images_per_class = get_images_per_class(dog_images_dir)

# Choose two images from each class
selected_images = choose_two_images_per_class(images_per_class)

# Convert color images to grayscale
grayscale_images = {}
for class_name, images in selected_images.items():
    grayscale_images[class_name] = convert_to_grayscale(images)

# Display grayscale images
for class_name, images in grayscale_images.items():
    plt.figure(figsize=(8, 4))
    for i, image in enumerate(images, start=1):
        plt.subplot(1, 2, i)
        plt.imshow(image, cmap='gray')
        plt.title(f'{class_name} - Image {i}')
        plt.axis('off')
    plt.show()


# In[29]:


import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import color

# Function to get the list of images for each class
def get_images_per_class(images_dir):
    images_per_class = {}
    for image_path in images_dir:
        class_name = os.path.basename(os.path.dirname(image_path))
        if class_name not in images_per_class:
            images_per_class[class_name] = []
        images_per_class[class_name].append(image_path)
    return images_per_class

# Function to choose two images from each class
def choose_two_images_per_class(images_per_class):
    selected_images = {}
    for class_name, images in images_per_class.items():
        selected_images[class_name] = images[:2]
    return selected_images

# Function to convert color images to grayscale
def convert_to_grayscale(images):
    grayscale_images = []
    for image_path in images:
        image = Image.open(image_path)
        grayscale_image = color.rgb2gray(np.array(image))
        grayscale_images.append(grayscale_image)
    return grayscale_images

# Function to plot images and their histograms
def plot_images_and_histograms(grayscale_images):
    for class_name, images in grayscale_images.items():
        plt.figure(figsize=(10, 5))
        for i, image in enumerate(images, start=1):
            plt.subplot(2, 4, i)
            plt.imshow(image, cmap='gray')
            plt.title(f'{class_name} - Image {i}')
            plt.axis('off')

            plt.subplot(2, 4, i+4)
            plt.hist(image.ravel(), bins=256, color='black')
            plt.title('Histogram')
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

# Path to the directory containing images
dog_images_dir = glob.glob(r"C:/Users/ruthv/DataMining/images/*/*")

# Get list of images per class
images_per_class = get_images_per_class(dog_images_dir)

# Choose two images from each class
selected_images = choose_two_images_per_class(images_per_class)

# Convert color images to grayscale
grayscale_images = {}
for class_name, images in selected_images.items():
    grayscale_images[class_name] = convert_to_grayscale(images)

# Plot grayscale images and histograms
plot_images_and_histograms(grayscale_images)


# In[23]:


import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import color, filters

# Function to get the list of images for each class
def get_images_per_class(images_dir):
    images_per_class = {}
    for image_path in images_dir:
        class_name = os.path.basename(os.path.dirname(image_path))
        if class_name not in images_per_class:
            images_per_class[class_name] = []
        images_per_class[class_name].append(image_path)
    return images_per_class

# Function to choose two images from each class
def choose_two_images_per_class(images_per_class):
    selected_images = {}
    for class_name, images in images_per_class.items():
        selected_images[class_name] = images[:2]
    return selected_images

# Function to convert color images to grayscale
def convert_to_grayscale(images):
    grayscale_images = []
    for image_path in images:
        image = Image.open(image_path)
        grayscale_image = color.rgb2gray(np.array(image))
        grayscale_images.append(grayscale_image)
    return grayscale_images

# Function to perform edge detection using Sobel filter
def detect_edges(images):
    edge_images = []
    for image in images:
        edge_image = filters.sobel(image)
        edge_images.append(edge_image)
    return edge_images

# Plot grayscale images and their corresponding edge-detected images
def plot_images_and_edges(grayscale_images, edge_images):
    plt.figure(figsize=(12, 6))
    for i, (gray_image, edge_image) in enumerate(zip(grayscale_images, edge_images), start=1):
        plt.subplot(2, 4, i)
        plt.imshow(gray_image, cmap='gray')
        plt.title(f'Grayscale Image {i}')
        plt.axis('off')

        plt.subplot(2, 4, i+4)
        plt.imshow(edge_image, cmap='gray')
        plt.title(f'Edge-Detected Image {i}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Path to the directory containing images
dog_images_dir = glob.glob(r"C:/Users/ruthv/DataMining/images/*/*")

# Get list of images per class
images_per_class = get_images_per_class(dog_images_dir)

# Choose two images from each class
selected_images = choose_two_images_per_class(images_per_class)

# Convert color images to grayscale
grayscale_images = {}
for class_name, images in selected_images.items():
    grayscale_images[class_name] = convert_to_grayscale(images)

# Perform edge detection using Sobel filter
edge_images = {}
for class_name, images in grayscale_images.items():
    edge_images[class_name] = detect_edges(images)

# Plot grayscale images and their corresponding edge-detected images
plot_images_and_edges(list(grayscale_images.values())[0], list(edge_images.values())[0])


# In[5]:


import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import color, filters
import glob

# Function to get the list of images for each class
def get_images_per_class(images_dir):
    images_per_class = {}
    for image_path in images_dir:
        class_name = os.path.basename(os.path.dirname(image_path))
        if class_name not in images_per_class:
            images_per_class[class_name] = []
        images_per_class[class_name].append(image_path)
    return images_per_class

# Function to choose two images from each class
def choose_two_images_per_class(images_per_class):
    selected_images = {}
    for class_name, images in images_per_class.items():
        selected_images[class_name] = images[:2]
    return selected_images

# Function to convert color images to grayscale
def convert_to_grayscale(images):
    grayscale_images = []
    for image_path in images:
        image = Image.open(image_path)
        grayscale_image = color.rgb2gray(np.array(image))
        grayscale_images.append(grayscale_image)
    return grayscale_images

# Function to perform edge detection using Sobel filter
def detect_edges(images):
    edge_images = []
    for image in images:
        edge_image = filters.sobel(image)
        edge_images.append(edge_image)
    return edge_images

# Plot grayscale images and their corresponding edge-detected images
def plot_images_and_edges(grayscale_images, edge_images):
    plt.figure(figsize=(12, 6))
    for i, (gray_image, edge_image) in enumerate(zip(grayscale_images, edge_images), start=1):
        plt.subplot(2, 4, i)
        plt.imshow(gray_image, cmap='gray')
        plt.title(f'Grayscale Image {i}')
        plt.axis('off')

        plt.subplot(2, 4, i+4)
        plt.imshow(edge_image, cmap='gray')
        plt.title(f'Edge-Detected Image {i}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Path to the directory containing images
dog_images_dir = glob.glob((r"C:/Users/ruthv/DataMining/images/*/*")

# Get list of images per class
images_per_class = get_images_per_class(dog_images_dir)

# Choose two images from each class
selected_images = choose_two_images_per_class(images_per_class)

# Convert color images to grayscale and perform edge detection
grayscale_images = {}
edge_images = {}
for class_name, images in selected_images.items():
    grayscale_images[class_name] = convert_to_grayscale(images)
    edge_images[class_name] = detect_edges(grayscale_images[class_name])

# Plot grayscale images and their corresponding edge-detected images
for class_name in grayscale_images.keys():
    plot_images_and_edges(grayscale_images[class_name], edge_images[class_name])


# In[37]:


import cv2
import numpy as np
from skimage import filters
import matplotlib.pyplot as plt

def angle(dx, dy):
    """Calculate the angles between horizontal and vertical operators."""
    return np.mod(np.arctan2(dy, dx), np.pi)

# Function to compute edge histograms
def compute_edge_histogram(image):
    sobel_h = filters.sobel_h(image)
    sobel_v = filters.sobel_v(image)
    angles = angle(sobel_h, sobel_v)
    hist, bin_edges = np.histogram(angles, bins=8, range=(0, np.pi))
    return hist

# Load one image from each class and convert them to grayscale
image_paths = [
    r"C:\Users\ruthv\Downloads\images\n02113799-standard_poodle\n02113799_911.jpg",
    r"C:\Users\ruthv\Downloads\images\n02107312-miniature_pinscher\n02107312_7528.jpg",
    r"C:\Users\ruthv\Downloads\images\n02094114-Norfolk_terrier\n02094114_981.jpg",
    r"C:\Users\ruthv\Downloads\images\n02096177-cairn\n02096177_8975.jpg"    # Add paths to other images here
]

grayscale_images = [cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) for image_path in image_paths]

# Compute edge histograms for each grayscale image
edge_histograms = [compute_edge_histogram(image) for image in grayscale_images]

# Plot edge histograms
plt.figure(figsize=(10, 6))
for i, hist in enumerate(edge_histograms):
    plt.plot(hist, label=f'Image {i+1}')
plt.xlabel('Angle Bins')
plt.ylabel('Frequency')
plt.title('Edge Histograms')
plt.legend()
plt.show()


# In[46]:


import cv2
import numpy as np
from skimage import filters, exposure
import matplotlib.pyplot as plt

def calculate_angles(dx, dy):
    """Calculate the angles between horizontal and vertical operators."""
    return np.mod(np.arctan2(dy, dx), np.pi)

def compute_edge_histogram(image):
    sobel_h = filters.sobel_h(image)
    sobel_v = filters.sobel_v(image)
    angles = calculate_angles(sobel_h, sobel_v)
    hist, bin_centers = exposure.histogram(image.ravel(), nbins=36, source_range='image')
    return hist

# Load one image from each class and convert them to grayscale
image_paths = [
    r"C:\Users\ruthv\Downloads\images\n02113799-standard_poodle\n02113799_911.jpg",
    r"C:\Users\ruthv\Downloads\images\n02107312-miniature_pinscher\n02107312_7528.jpg",
    r"C:\Users\ruthv\Downloads\images\n02094114-Norfolk_terrier\n02094114_981.jpg",
    r"C:\Users\ruthv\Downloads\images\n02096177-cairn\n02096177_8975.jpg"
    # Add paths to other images here
]

grayscale_images = [cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) for image_path in image_paths]

# Compute edge histograms for each grayscale image
edge_histograms = [compute_edge_histogram(image) for image in grayscale_images]

# Plot edge histograms
plt.figure(figsize=(10, 6))
for i, current_hist in enumerate(edge_histograms):
    plt.plot(current_hist, label=f'Image {i+1}')
plt.xlabel('Bins')
plt.ylabel('Frequency')
plt.title('Edge Histograms')
plt.legend()
plt.show()


# In[45]:


import cv2
import numpy as np
from skimage import filters, exposure
import matplotlib.pyplot as plt

def calculate_angles(dx, dy):
    """Calculate the angles between horizontal and vertical operators."""
    return np.mod(np.arctan2(dy, dx), np.pi)

def compute_edge_histogram(image):
    sobel_h = filters.sobel_h(image)
    sobel_v = filters.sobel_v(image)
    angles = calculate_angles(sobel_h, sobel_v)
    hist, bin_centers = exposure.histogram(image.ravel(), nbins=36, source_range='image')
    return hist, bin_centers

# Load one image from each class and convert them to grayscale
image_paths = [
    r"C:\Users\ruthv\Downloads\images\n02113799-standard_poodle\n02113799_911.jpg",
    r"C:\Users\ruthv\Downloads\images\n02107312-miniature_pinscher\n02107312_7528.jpg",
    r"C:\Users\ruthv\Downloads\images\n02094114-Norfolk_terrier\n02094114_981.jpg",
    r"C:\Users\ruthv\Downloads\images\n02096177-cairn\n02096177_8975.jpg"
    # Add paths to other images here
]
grayscale_images = [cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) for image_path in image_paths]

# Compute edge histograms for each grayscale image
edge_histograms = [compute_edge_histogram(image) for image in grayscale_images]

# Plot images with their corresponding edge histograms
fig, axes = plt.subplots(len(grayscale_images), 2, figsize=(10, 6 * len(grayscale_images)))

for i, (current_image, (current_hist, current_bin_centers)) in enumerate(zip(grayscale_images, edge_histograms)):
    # Plot current image
    axes[i, 0].imshow(current_image, cmap='gray')
    axes[i, 0].set_title('Image')
    
    # Plot current edge histogram
    axes[i, 1].bar(current_bin_centers, current_hist, width=np.pi / 18)
    axes[i, 1].set_xlabel('Bins')
    axes[i, 1].set_ylabel('Pixel Count')
    axes[i, 1].set_title('Edge Histogram')

plt.tight_layout()
plt.show()


# In[13]:


#for same class
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

image1_path = "C:/Users/ruthv/Downloads/histogram/n02107312_7.jpg"
image2_path = "C:/Users/ruthv/Downloads/histogram/n02107312_5059.jpg"

hist1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
hist2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

# Resize both histograms to the same shape
desired_shape = (500, 323)  # Adjust to the desired shape
hist1 = cv2.resize(hist1, desired_shape)
hist2 = cv2.resize(hist2, desired_shape)

# Flatten the histograms
hist1_flat = hist1.flatten().reshape(1, -1)
hist2_flat = hist2.flatten().reshape(1, -1)

# Euclidean Distance
euclidean_distance = np.linalg.norm(hist1_flat - hist2_flat)

# Manhattan Distance
manhattan_distance = np.sum(np.abs(hist1_flat - hist2_flat))

# Cosine Similarity
cosine_similarity_score = cosine_similarity(hist1_flat, hist2_flat)[0][0]
cosine_distance = 1 - cosine_similarity_score
i
print(f"Euclidean Distance: {euclidean_distance}")
print(f"Manhattan Distance: {manhattan_distance}")
print(f"Cosine Distance: {cosine_distance}")


# In[40]:


get_ipython().system('pip install opencv-python')



# In[17]:


pip install opencv-python


# In[19]:


pip install numpy


# In[44]:


import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

image_path1 = "C:/Users/ruthv/Downloads/histogram/n02107312_7.jpg"
image_path2 = "C:/Users/ruthv/Downloads/histogram/n02107312_5059.jpg"

hist1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
hist2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

# Resize both histograms to the same shape
desired_shape = (500, 323)  # Adjust to the desired shape
hist1 = cv2.resize(hist1, desired_shape)
hist2 = cv2.resize(hist2, desired_shape)

# Flatten the histograms
hist1_flat = hist1.flatten().reshape(1, -1)
hist2_flat = hist2.flatten().reshape(1, -1)

# Euclidean Distance
euclidean_distance = np.linalg.norm(hist1_flat - hist2_flat)

# Manhattan Distance
manhattan_distance = np.sum(np.abs(hist1_flat - hist2_flat))

# Cosine Similarity
cosine_similarity_score = cosine_similarity(hist1_flat, hist2_flat)[0][0]
cosine_distance = 1 - cosine_similarity_score

print(f"Euclidean Distance: {euclidean_distance}")
print(f"Manhattan Distance: {manhattan_distance}")
print(f"Cosine Distance: {cosine_distance}")


# In[43]:


import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

image_path1 = "C:/Users/ruthv/Downloads/histogram/n02107312_5059.jpg"
image_path2 = "C:/Users/ruthv/Downloads/histogram/n02113799_1536.jpg"

hist1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
hist2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

# Resize both histograms to the same shape
desired_shape = (500, 323)  # Adjust to the desired shape
hist1 = cv2.resize(hist1, desired_shape)
hist2 = cv2.resize(hist2, desired_shape)

# Flatten the histograms
hist1_flat = hist1.flatten().reshape(1, -1)
hist2_flat = hist2.flatten().reshape(1, -1)

# Euclidean Distance
euclidean_distance = np.linalg.norm(hist1_flat - hist2_flat)

# Manhattan Distance
manhattan_distance = np.sum(np.abs(hist1_flat - hist2_flat))

# Cosine Similarity
cosine_similarity_score = cosine_similarity(hist1_flat, hist2_flat)[0][0]
cosine_distance = 1 - cosine_similarity_score

print(f"Euclidean Distance: {euclidean_distance}")
print(f"Manhattan Distance: {manhattan_distance}")
print(f"Cosine Distance: {cosine_distance}")


# In[33]:


import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure

def calculate_and_display_hog(image_path):
    grayscale_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    hog_features, hog_image = hog(grayscale_image, visualize=True)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(grayscale_image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(hog_image, cmap='gray')
    plt.title('HOG Descriptors')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

new_image_path = "C:\\Users\\ruthv\\Downloads\\images\\n02094114-Norfolk_terrier\\n02094114_1232.jpg"

calculate_and_display_hog(new_image_path)


# In[42]:


import cv2
import os
import numpy as np
from skimage import filters, exposure
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def calculate_edge_histograms(image_paths):
    edge_histograms = []
    for img_path in image_paths:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        angles = (filters.sobel_v(img), filters.sobel_h(img))
        hist_v, _ = exposure.histogram(angles[0], nbins=36)
        hist_h, _ = exposure.histogram(angles[1], nbins=36)
        edge_histograms.append(np.concatenate((hist_v, hist_h)))
    return edge_histograms

class_directories = {
    'class1': r"C:\Users\ruthv\Downloads\images\n02094114-Norfolk_terrier",
    'class2': r"C:\Users\ruthv\Downloads\images\n02107312-miniature_pinscher"
}

selected_images = []
for class_label, directory in class_directories.items():
    image_files = os.listdir(directory)
    class_images = [os.path.join(directory, img) for img in image_files]
    selected_images.extend(class_images)

edge_histograms = calculate_edge_histograms(selected_images)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(edge_histograms)

num_class1_images = len(os.listdir(class_directories['class1']))
plt.scatter(pca_result[:num_class1_images, 0], pca_result[:num_class1_images, 1], c='green', label='class1')
plt.scatter(pca_result[num_class1_images:, 0], pca_result[num_class1_images:, 1], c='purple', label='class2')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Dimensionality Reduction')
plt.legend()
plt.show()


# In[ ]:




