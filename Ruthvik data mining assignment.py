#!/usr/bin/env python
# coding: utf-8

# In[16]:


import os
import glob
from PIL import Image
from pathlib import Path
import xml.etree.ElementTree as ET

# Update the paths to use raw strings or double backslashes
dog_images_dir = glob.glob(r"C:\Users\ruthv\DataMining\images\*\*")
annotations_dir = glob.glob(r"C:\Users\ruthv\DataMining\Annotations\*\*")
cropped_images_dir = r"C:\Users\ruthv\DataMining\cropped"

def get_bounding_boxes(annot):
    xml = annot
    tree = ET.parse(xml)
    root = tree.getroot()
    objects = root.findall('object')
    bbox = []
    for o in objects:
        bndbox = o.find('bndbox' )
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        bbox.append((xmin, ymin, xmax, ymax))
    return bbox

def get_image(annot):
    img_path = r"C:\Users\ruthv\DataMining\images"
    file = annot.split('\\')
    img_filename = os.path.join(img_path, file[-2], file[-1] + '.jpg')
    return img_filename

# Check if both dog_images_dir and annotations_dir are not empty and have the same length
if dog_images_dir and annotations_dir and len(dog_images_dir) == len(annotations_dir):
    print(len(dog_images_dir))
    for dog_dir, annot_dir in zip(dog_images_dir, annotations_dir):
        bbox = get_bounding_boxes(annot_dir)
        dog = get_image(annot_dir)
        try:
            im = Image.open(dog)
        except Exception as e:
            print(f"Error opening image file: {dog}")
            print(f"Error details: {e}")
            continue

        print(im)
        for j, box in enumerate(bbox):
            im2 = im.crop(box)
            im2 = im2.resize((128, 128), Image.LANCZOS)
            new_path = dog.replace('images', 'cropped')
            new_path = new_path.replace('.jpg', '-' + str(j) + '.jpg')
            head, tail = os.path.split(new_path)
            Path(head).mkdir(parents=True, exist_ok=True)
            im2.save(new_path)


# In[1]:


import os
import matplotlib.pyplot as plt
from PIL import Image
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

# Path to the directory containing images
dog_images_dir = glob.glob(r"C:/Users/ruthv/DataMining/images/*/*")

# Get list of images per class
images_per_class = get_images_per_class(dog_images_dir)

# Choose two images from each class
selected_images = choose_two_images_per_class(images_per_class)

# Display color images
for class_name, images in selected_images.items():
    plt.figure(figsize=(8, 4))
    for i, image_path in enumerate(images, start=1):
        image = Image.open(image_path)
        plt.subplot(1, 2, i)
        plt.imshow(image)
        plt.title(f'{class_name} - Image {i}')
        plt.axis('off')
    plt.show()


# In[19]:


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


# In[3]:


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


# In[11]:


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
dog_images_dir = glob.glob(r"C:/Users/ruthv/DataMining/images/*/*")

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


# In[6]:


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


# In[12]:


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


# In[5]:


import cv2
import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt

def compute_edge_histogram(image):
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Compute the edges for each channel
    edges_h = cv2.Sobel(hsv_image[:, :, 0], cv2.CV_64F, 1, 0, ksize=3)
    edges_s = cv2.Sobel(hsv_image[:, :, 1], cv2.CV_64F, 1, 0, ksize=3)
    edges_v = cv2.Sobel(hsv_image[:, :, 2], cv2.CV_64F, 1, 0, ksize=3)
    
    # Calculate angles for each channel (you might need to adjust this based on your requirements)
    angles_h = np.mod(np.arctan2(edges_h, 1), np.pi)
    angles_s = np.mod(np.arctan2(edges_s, 1), np.pi)
    angles_v = np.mod(np.arctan2(edges_v, 1), np.pi)
    
    # Concatenate the angles from all channels
    angles = np.concatenate([angles_h.ravel(), angles_s.ravel(), angles_v.ravel()])
    
    # Compute the histogram
    hist, bin_centers = exposure.histogram(angles, nbins=36)
    
    return hist, bin_centers

# Load one image from each class
image_paths = [
    r"C:\Users\ruthv\Downloads\images\n02113799-standard_poodle\n02113799_911.jpg",
    r"C:\Users\ruthv\Downloads\images\n02107312-miniature_pinscher\n02107312_7528.jpg",
    r"C:\Users\ruthv\Downloads\images\n02094114-Norfolk_terrier\n02094114_981.jpg",
    r"C:\Users\ruthv\Downloads\images\n02096177-cairn\n02096177_8975.jpg"
    # Add paths to other images here
]
class_labels = ["Standard Poodle", "Miniature Pinscher", "Norfolk Terrier", "Cairn"]  # Add corresponding class labels

color_images = [cv2.imread(image_path) for image_path in image_paths]

# Compute edge histograms for each color image
edge_histograms = [compute_edge_histogram(image) for image in color_images]

# Plot images with their corresponding edge histograms
fig, axes = plt.subplots(len(color_images), 2, figsize=(10, 6 * len(color_images)))

for i, (current_image, (current_hist, current_bin_centers)) in enumerate(zip(color_images, edge_histograms)):
    # Plot current color image
    axes[i, 0].imshow(cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB))
    axes[i, 0].set_title(f'{class_labels[i]} - Image')
    
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


# In[ ]:


get_ipython().system('pip install opencv-python')



# In[ ]:


pip install opencv-python


# In[ ]:


pip install numpy


# In[14]:


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


# In[15]:


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


# In[6]:


import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure

def calculate_and_display_hog(image_path):
    # Read the image in color
    color_image = cv2.imread(image_path)

    # Convert the color image to grayscale for calculating HOG features
    grayscale_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # Calculate HOG features and get a visualization of the HOG descriptors
    hog_features, hog_image = hog(grayscale_image, visualize=True)

    # Display the original color image, grayscale image, and HOG descriptors
    plt.figure(figsize=(12, 4))
    
    # Plot the original color image
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
    plt.title('Original Color Image')
    plt.axis('off')
    
    # Plot the grayscale image
    plt.subplot(1, 3, 2)
    plt.imshow(grayscale_image, cmap='gray')
    plt.title('Grayscale Image')
    plt.axis('off')
    
    # Plot the HOG descriptors
    plt.subplot(1, 3, 3)
    plt.imshow(hog_image, cmap='gray')
    plt.title('HOG Descriptors')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Example usage with a specific image path
new_image_path = "C:\\Users\\ruthv\\Downloads\\images\\n02094114-Norfolk_terrier\\n02094114_1232.jpg"
calculate_and_display_hog(new_image_path)


# In[17]:


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

