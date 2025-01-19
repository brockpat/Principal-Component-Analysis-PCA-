# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 18:05:42 2024

@author: Patrick
"""

#https://www.tejwin.com/en/insight/pca-feature-portfolio/

import numpy as np
from sklearn.decomposition import PCA
import copy
import matplotlib.pyplot as plt
import copy

#------- Generate Data
#Setting the seed
rng = np.random.default_rng(489481156)

#Getting x
x = rng.uniform(low=-3, high=3, size=150)
#Add some more data close to zero
x = np.append(x,np.array([-0.2,-0.17,0.13,0.07]))
#Draw an error components
eps = rng.normal(0,0.8,size=len(x))
#Construct y
y = 0.5*x+0.4*eps
#Generate data frame of data
X = np.array([x,y]).T

#------- Compute PCA by hand
#Compute the covariance matrix of the data
Sigma = np.cov(X[:,0],X[:,1],rowvar = False)

#Compute eigenvalues and eigenvectors of covariance matrix
eigenvalue, eigenvectors = np.linalg.eig(Sigma) #eigenvectors are column wise
#i-th row of eigenvectors is alpha_i

#------- Compute PCA by package
# Perform PCA
pca = PCA(n_components=2)
pca.fit(X)

# Get the principal components
components = pca.components_ #1st row is alpha_1, 2nd row is alpha_2 

# Explained variance ratio
print("Explained variance ratio:", pca.explained_variance_ratio_)

#Check that linear combinations are actually uncorrelated
pca1 = components[0,0]*X[:,0] + components[0,1]*X[:,1]
pca2 = components[1,0]*X[:,0] + components[1,1]*X[:,1]
np.cov(pca1,pca2)

#Check explained variance ratio
np.cov(pca1,pca2)[0,0]/ (np.cov(pca1,pca2)[0,0] + np.cov(pca1,pca2)[1,1])

#------- Visualisation
# Plot the scatter plot of the data
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], alpha=0.6, label="Data Points")

# Plot the first principal component
component = components[0,:]
plt.arrow(
    0, 0,  # Start of the arrow (mean of the data)
    (-1)*component[0] * 3, (-1)*component[1] * 3,  # Scale the component for visibility
    head_width=0.2, head_length=0.3, color='orange', label='PC1'
)
    
# Plot the second principal component
component = components[1,:]
plt.arrow(
    0, 0,  # Start of the arrow (mean of the data)
    -component[0], -component[1],  # Scale the component for visibility
    head_width=0.15, head_length=0.3, color='green', label='PC2'
)

# Formatting the plot
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.title("2D Scatter Plot with Principal Components")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
#%% Further computations

#------- Matrix of eigenvectors
#Construct the Matrix of eigenvectors (column wise) and flip the orientation as depicted in the plots
A = -components.T #Note that eigenvectors (columns) have unit length so no stretching occurs


#Compute the inverse
A_inv = np.linalg.inv(A) #Notice that A_inv = A.T

#------- Change of Basis
#Rotatating the plane such that PCs are the new basis vectors
e1_new = A_inv @ np.array([1,0]) #where the first old basis vector goes
e2_new = A_inv @ np.array([0,1]) #where the second old basis vector goes

#Rotating the data points (maintaining that X_PCA is of dimension Nx2)
X_PCA = (A_inv @ X.T).T
#This has the following underpinning: 
    # x_pca1 = x[i,:]@alpha1/(alpha1@alpha1) * (1,0) (first coordinate)
    # x_pca2 = x[i,:]@alpha2/(alpha2@alpha2) * (0,1) (second coordinate)
    # notice that alpha_i @ alpha_i is equal to 1 
    # the principal components (alpha_1 and alpha_2) are the new basis vectors (1,0) and (0,1)
    # (this is what the matrix A does)

#------- Orthogonal Projection
P0 = copy.deepcopy(X_PCA)
#Project rotated data orthogonally onto PC1
P0[:,1] = 0 #As PCs new basis vectors, this simply means setting the second component to zero

#Rotate the orthogonal projection back to the normal data
P = (A @ P0.T).T 
#This has the following underpinning
    #projection onto PC1: x[i,:]@alpha_1/(alpha_1@alpha_1)* alpha_1 (notice that alpha_1@alpha_1 = 1)
#%% Generate Exports for Tikz plots

#Scatter Plot untransformed data
f = open("file.txt", "w")
for i in range(0,len(X)):
    f.write(str(round(X[i,0],4)) + "/" + str(round(X[i,1],4)) + "," + "\n")
f.close()

#Scatter plot rotated data
f = open("file.txt", "w")
for i in range(0,len(X_PCA)):
    f.write(str(round(X_PCA[i,0],4)) + "/" + str(round(X_PCA[i,1],4)) + "," + "\n")
f.close()

#Drawing the orthogonal projection from unroated data onto PC1
f = open("file.txt", "w")
for i in range(0,len(P)):
    f.write("\\draw[dotted, thick] (" + 
            str(round(X[i,0],4)) + "," + str(round(X[i,1],4)) + ")" 
            + "--" + 
            "(" + 
            str(round(P[i,0],4)) + "," + str(round(P[i,1],4)) + ")" + 
            ";" + "\n")
f.close()


