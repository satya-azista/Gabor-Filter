#satya sai thota  sai sd .kgo'ih

import cv2
import numpy as np
from sklearn.cluster import KMeans
from math import sqrt

#Gabor images Genrated depended on wavelength(Format-array) , oreintation(Format-array) , sigma , gamma , phi values
def gabor_image_generator(image,sigma,theta,wavelength,gamma,phi,kernel_size):
  stack_image=[]
  gab=[]
  for t in theta:
    for w in wavelength:
      gabor_kernel=cv2.gaborfilter((kernel_size,kernel_size),sigma,t,w,gamma,phi)
      gabor_image=cv2.filter2d(image,cv2.CV2_32F,gabor_kernel)
      gab.append(gabor_image)
  stack_image=np.dstack(gab)
  return stack_image

#mean array generation of specifeid regions
def mean_array_generator(stack_image,cordinates):
    mean_array=[]
    for i in range(len(cordinates)):
        arr=[]
        for j in range(len(stack_image)):
            start_col,end_col=cordinates[i][0],cordinates[i][2]
            start_row,end_row=cordinates[i][1],cordinates[i][3]
            region = [row[start_col:end_col] for row in stack_image[j][start_row:end_row]]
            flattened_region = [element for row in region for element in row]
            mean_values=np.mean(flattened_region)
            arr.append(mean_values)
        mean_array.append[arr]
    return mean_array

#Eucliad arrray generation by mean and stacked_images array
def euclid_array_generator(mean_array,image_stack):
    euclid_distances_image= np.zeros((image_stack.shape[0], image_stack.shape[1]))
    for i in range(image_stack.shape[0]):
        for j in range(image_stack.shape[1]):
            val = 0
            for k in range(image_stack.shape[2]):
                val += (mean_array[k] - image_stack[i, j, k]) ** 2
            euclid_distances_image[i][j]=sqrt(val)
    return euclid_distances_image

#applying Kmeans to euclid image
def kmeans_light(data, K):
    N, dim = data.shape
    stop_iter = 0.05
    codebook = data[np.random.choice(N, K, replace=False), :]
    improved_ratio = np.inf
    distortion = np.inf
    iter_count = 0

    while True:
        # Calculate Euclidean distances between each sample and each codeword
        d = eucdist2(data, codebook)
        # Assign each sample to the nearest codeword (centroid)
        data_near_cluster_dist, data_cluster = np.min(d, axis=1), np.argmin(d, axis=1)
        
        # Distortion. If centroids are unchanged, distortion is also unchanged.
        old_distortion = distortion
        distortion = np.mean(data_near_cluster_dist)
        improved_ratio = 1 - (distortion / old_distortion)
        print(f'{iter_count}: improved ratio = {improved_ratio}')
        iter_count += 1

        if improved_ratio <= stop_iter:
            break

        # Renew codebook
        for i in range(K):
            # Get the indices of samples which were clustered into cluster i
            idx = data_cluster == i
            # Calculate centroid of each cluster, and replace codebook
            codebook[i, :] = np.mean(data[idx, :], axis=0)

    return data_cluster, codebook

def eucdist2(X, Y):
    U = ~np.isnan(Y)
    Y[~U] = 0
    V = ~np.isnan(X)
    X[~V] = 0
    d = np.abs(X**2 @ U.T + V @ Y.T**2 - 2 * X @ Y.T)
    return d
  
if __name__== "__main__":
  file_path=r"C:\Users\Training\Desktop\delhi2.jpg"
  image=cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)
  sigma=1
  gamma=1
  phi=0
  theta=np.arange(0,np.pi,np.pi/6)
  r,c=image.shape
  j=(2**(np.arange((0,np.log2(c/8)+1))-0.5))/c
  f=np.concatenate([0.25-j,0.25+j])
  wavlength=1/f
  k_size=3
  stacked_image=gabor_image_generator(image,sigma,theta,wavelength,gamma,phi,k_size)
  coordinates=[[132,13,173,54],[870,601,911,642],[900,1361,941,1402],[1250,129,,1770,1811],[1728,303,1769,344],[2402,2443,1544,1595],[3262,809,3303,850],[3424,3465,1084,1125]]
  mean_array=mean_array_generator(stacked_image,coordinates)
  euclid_array=euclid_array_generator(mean_array,stacked_image)
  # Assuming 'data' is a numpy array with your input data
  # 'K' is the number of clusters you want to form
  k=2
  dataCluster, codebook = kmeans_light(euclid_array, k)
  

  
    
