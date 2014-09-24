
# coding: utf-8

# # PCA and Facial Recognition (sklearn edition)
# 
# ## Author: Ryan Quan
# ## Date: September 23, 2014
# 
# This is a Python rendition of principal component analysis in the context of facial recognition using the Extended Yale Faces Database B which you can download [here](http://vision.ucsd.edu/content/extended-yale-face-database-b-b). Originally done in R, this was written in order to experiment with the `sklearn` library. If you have any questions about this notebook, please do not hesitate to contact me at [ryan.quan08@gmail.com](mailto:ryan.quan08@gmail.com).
# 

# # Setup

# In[1]:

get_ipython().magic(u'matplotlib inline')
import numpy as np
import os
import matplotlib.pyplot as plt
import math
from PIL import Image

get_ipython().system(u'pwd')
data_dir = "/Users/Quan/GitHub/sklearn-practice/CroppedYale"
os.chdir(data_dir)
get_ipython().system(u'ls')


# # Converting .pgm to .png
# If your files are not already in the .png format, you can use this block of code to convert the file using the UNIX shell. Credits to UCSD_Big_Data for this block of code.

# In[2]:

# converting from svg to png
# from glob import glob

# %cd $data_dir

# files=glob('yaleB*/*.pgm')
# print 'number of files is',len(files)
# count=0
# for f in files:
#     new_f=f[:-3]+'png'
#     !convert $f $new_f
#     count += 1
#     if count % 100==0:
#         print count,f,new_f


# This `image_grid_` function reshapes the array into its original dimensions and plots the image in a grid. The `col` parameter allows you to specify the number of images in each row.

# In[3]:

def image_grid(D,H,W,cols=10,scale=1):
    """ display a grid of images
        H,W: Height and width of the images
        cols: number of columns = number of images in each row
        scale: 1 to fill screen
    """
    n = np.shape(D)[0]
    rows = int(math.ceil((n+0.0)/cols))
    fig = plt.figure(1,figsize=[scale*20.0/H*W,scale*20.0/cols*rows],dpi=300)
    for i in range(n):
        plt.subplot(rows,cols,i+1)
        fig=plt.imshow(np.reshape(D[i,:],[H,W]), cmap = plt.get_cmap("gray"))
        plt.axis('off')


# # Loading the Data
# `create_filenames` allows the users to specify the current working directory where the `CroppedYale` folder resides and the image view for the subjects.

# In[4]:

def create_filenames(data_dir, view_list):
    # loads the pictures into a list
    # data_dir: the CroppedYale folder
    # view_list: the views you wish to grab
    dir_list = os.listdir(data_dir)
    file_list = []
    for dir in dir_list:
        for view in view_list:
            filename = "%s/%s_%s.png" % (dir, dir, view)
            file_list.append(filename)
    return(file_list)


view_list = ['P00A+000E+00', 'P00A+005E+10' , 'P00A+005E-10' , 'P00A+010E+00']

file_list = create_filenames(data_dir, view_list)
len(file_list)


# ## Our "Faces" Data

# In[5]:

# open image
im = Image.open(file_list[0]).convert("L")
# get original dimensions
H,W = np.shape(im)
print 'shape=',(H,W)

im_number = len(file_list)
# fill array with rows as image
# and columns as pixels
arr = np.zeros([im_number,H*W])

for i in range(im_number):
    im = Image.open(file_list[i]).convert("L")
    arr[i,:] = np.reshape(np.asarray(im),[1,H*W])

image_grid(arr,H,W)


# ## The Mean Face

# In[6]:

# let's find the mean_image
mean_image = np.mean(arr, axis=0)

plt.imshow(np.reshape(mean_image,[H,W]), cmap = plt.get_cmap("gray"))
plt.figure()
plt.hist(mean_image,bins=100);


# In[7]:

# centering the data (subtract mean face)
arr_norm = np.zeros([im_number, H*W])
arr_norm = arr - mean_image


# In[8]:

# plot the first 10 normalized faces
image_grid(arr_norm[:10,:],H,W)


# # Principal Component Analysis

# In[9]:

from sklearn.decomposition.pca import PCA


# In[10]:

pca = PCA()
pca.fit(arr_norm)


# ## Scree Plot

# In[11]:

# Let's make a scree plot
pve = pca.explained_variance_ratio_
pve.shape
plt.plot(range(len(pve)), pve)
plt.title("Scree Plot")
plt.ylabel("Proportion of Variance Explained")
plt.xlabel("Principal Component Number")


# ## Eigenfaces
# 
# The eigenvectors of the variance-covariance matrix of our "face" data represent the so-called "eigenfaces". They represent the direction of greatest variability in our "face space". We plot the first 9 eigenfaces here.

# In[12]:

# eigenfaces
eigenfaces = pca.components_
image_grid(eigenfaces[:9,:], H, W, cols=3)


# In[13]:

img_idx = file_list.index('yaleB01/yaleB01_P00A+010E+00.png')
loadings = pca.components_
n_components = loadings.shape[0]
scores = np.dot(arr_norm[:,:], loadings[:,:].T)

img_proj = []
for n in range(n_components):
    proj = np.dot(scores[img_idx, n], loadings[n,:])
    img_proj.append(proj)
len(img_proj)


# In[14]:

faces = mean_image
face_list = []
face_list.append(mean_image)
for i in range(len(img_proj)):
    faces = np.add(faces, img_proj[i])
    face_list.append(faces)

len(face_list)


# In[15]:

face_arr = np.asarray(face_list)
face_arr.shape


# ## Reconstructed Face (Adding 1 Base)

# In[16]:

image_grid(face_arr[:25], H, W, cols=5)


# ## Reconstructed Face (Adding 5 Base)

# In[17]:

image_grid(face_arr[range(0, 121, 5)], H, W, cols=5)


# # Reconstructing a Face from Principal Components
# 
# In this scenario, we remove a subject from the "face data" run another PCA without that subject in the training set. We then reconstruct the subject's face using the new principal components to see how similar the reconstructed face looks to the original image.

# In[18]:

# getting the index of the subject
sub_idx = [i for i, s in enumerate(file_list) if "yaleB05" in s]
print sub_idx


# In[19]:

face_idx = file_list.index("yaleB05/yaleB05_P00A+010E+00.png")
print face_idx


# ## Target Face

# In[20]:

# plot target face
image_grid(arr[19:20], H, W)


# In[21]:

# remove subject from array
arr_new = np.zeros([len(file_list), H*W])

for i in range(len(file_list)):
    im = Image.open(file_list[i]).convert("L")
    arr_new[i,:] = np.reshape(np.asarray(im),[1, H*W])

target_face = arr_new[19,]
    
arr_new = np.delete(arr_new, sub_idx, axis = 0)
arr_new.shape


# In[22]:

target_face.shape


# ## Centered Face

# In[23]:

mean_face = np.mean(arr_new, axis = 0)

centered_face = target_face - mean_face

plt.imshow(np.reshape(centered_face,[H,W]), cmap = plt.get_cmap("gray"))
plt.figure()
plt.hist(centered_face,bins=100);


# In[24]:

arr_norm = arr_new - mean_face
pca.fit(arr_new)


# In[25]:

loadings = pca.components_
n_components = loadings.shape[0]
scores = np.dot(centered_face, loadings.T)
reconstruct = np.dot(scores, loadings)

reconstruct.shape


# ## Reconstructed Face

# In[26]:

plt.imshow(np.reshape(reconstruct, [H,W]), cmap = plt.get_cmap("gray"))
plt.figure()

