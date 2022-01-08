# # Imports
# from sklearn.datasets import make_blobs
# import numpy as np
# import matplotlib.pyplot as plt
# # Generate 2D data points
# dd=int(input())


# # Convert the data points into a pandas DataFrame
# import pandas as pd

# # Generate indicators for the data points
# obj_names = []
# for i in range(1, 11):
#     obj = "Object " + str(i)
#     obj_names.append(obj)
# X=[]
# Z=[]
# for zz in range(0,dd):
#     kkk=int(input())
#     X.append(kkk)
# for zz in range(0,dd):
#     ggg=int(input())
#     Z.append(ggg)
# # Create a pandas DataFrame with the names and (x, y) coordinates
# data = pd.DataFrame({
    
#     'X_value': X,
#     'Y_value': Z
# })

# # Preview the data
# print(data.head())
# # Initialize the centroids
# c1 = (2, 10)
# c2 = (5, 8)
# c3 = (1, 2)
# # A helper function to calculate the Euclidean diatance between the data 
# # points and the centroids

# def calculate_distance(centroid, X, Y):
#     distances = []
        
#     # Unpack the x and y coordinates of the centroid
#     c_x, c_y = centroid
        
#     # Iterate over the data points and calculate the distance using the           # given formula
#     for x, y in list(zip(X, Y)):
#         root_diff_x = (x - c_x) ** 2
#         root_diff_y = (y - c_y) ** 2
#         distance = np.sqrt(root_diff_x + root_diff_y)
#         distances.append(distance)
        
#     return distances
#     # Calculate the distance and assign them to the DataFrame accordingly
# data['C1_Distance'] = calculate_distance(c1, data.X_value, data.Y_value)
# data['C2_Distance'] = calculate_distance(c2, data.X_value, data.Y_value)
# data['C3_Distance'] = calculate_distance(c3, data.X_value, data.Y_value)

# # Preview the data
# print(data.head())
# # Get the minimum distance centroids
# data['Cluster'] = data[['C1_Distance', 'C2_Distance', 'C3_Distance']].apply(np.argmin, axis =1)
    
#     # Map the centroids accordingly and rename them
# data['Cluster'] = data['Cluster'].map({'C1_Distance': 'C1', 'C2_Distance': 'C2', 'C3_Distance': 'C3'})
    
#     # Get a preview of the data
# print(data.head(10))
# # Calculate the coordinates of the new centroid from cluster 1
# x_new_centroid1 = data[data['Cluster']=='C1']['X_value'].mean()
# y_new_centroid1 = data[data['Cluster']=='C1']['Y_value'].mean()

# # Calculate the coordinates of the new centroid from cluster 2
# x_new_centroid2 = data[data['Cluster']=='C3']['X_value'].mean()
# y_new_centroid2 = data[data['Cluster']=='C3']['Y_value'].mean()

# # Calculate the coordinates of the new centroid from cluster 1
# x_new_centroid1 = data[data['Cluster']=='C1']['X_value'].mean()
# y_new_centroid1 = data[data['Cluster']=='C1']['Y_value'].mean()

# # Calculate the coordinates of the new centroid from cluster 2
# x_new_centroid2 = data[data['Cluster']=='C3']['X_value'].mean()
# y_new_centroid2 = data[data['Cluster']=='C3']['Y_value'].mean()

# # Print the coordinates of the new centroids
# print('Centroid 1 ({}, {})'.format(x_new_centroid1, y_new_centroid1))
# print('Centroid 2 ({}, {})'.format(x_new_centroid2, y_new_centroid2))
# # # Using scikit-learn to perform K-Means clustering
# # from sklearn.cluster import KMeans
    
# # # Specify the number of clusters (3) and fit the data X
# # kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
# # # Get the cluster centroids
# # print(kmeans.cluster_centers_)
    
# # # Get the cluster labels
# # print(kmeans.labels_)



# # # Calculate silhouette_score
# # from sklearn.metrics import silhouette_score

# # print(silhouette_score(X, kmeans.labels_))





# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# # using the make_blobs dataset
# from sklearn.datasets import make_blobs
# dd=int(input())
# # setting the number of training examples
# X=[]
# Y=[]
# for zzz in range(0,dd):
#     www=int(input())
#     X.append(www)
# for ggg in range(0,dd):
#     lll=int(input())
#     Y.append(lll)
# m=len(X)
# n=len(Y)
# n_iter=50
# # computing the initial centroids randomly
# K=3
# import random

# # creating an empty centroid array
# centroids=np.array([])

# # creating 5 random centroids
# for k in range(K):
#     centroids=np.c_[centroids,X[random.randint(0,m-1)]]
# output={}

# # creating an empty array
# euclid=np.array([])

# # finding distance between for each centroid
# for k in range(K):
#        dist=np.sum((X-centroids[:,k])**2,axis=1)
#        euclid=np.c_[euclid,dist]

# # storing the minimum value we have computed
# minimum=np.argmin(euclid,axis=1)+1
# # computing the mean of separated clusters
# cent={}
# for k in range(K):
#     cent[k+1]=np.array([]).reshape(2,0)

# # assigning of clusters to points
# for k in range(m):
#     cent[minimum[k]]=np.c_[cent[minimum[k]],X[k]]
# for k in range(K):
#     cent[k+1]=cent[k+1].T

# # computing mean and updating it
# for k in range(K):
#      centroids[:,k]=np.mean(cent[k+1],axis=0)
# # repeating the above steps again and again
# for i in range(n_iter):
#       euclid=np.array([]).reshape(m,0)
#       for k in range(K):
#           dist=np.sum((X-centroids[:,k])**2,axis=1)
#           euclid=np.c_[euclid,dist]
#       C=np.argmin(euclid,axis=1)+1
#       cent={}
#       for k in range(K):
#            cent[k+1]=np.array([]).reshape(2,0)
#       for k in range(m):
#            cent[C[k]]=np.c_[cent[C[k]],X[k]]
#       for k in range(K):
#            cent[k+1]=cent[k+1].T
#       for k in range(K):
#            centroids[:,k]=np.mean(cent[k+1],axis=0)
#       final=cent
# for k in range(K):
#     print(centroids[:,k])

import numpy as np
X=np.array([[2,10],[2,5],[8,4],[5,8],[7,5],[6,4],[1,2],[4,9]])
# Number of training examples
m=X.shape[0]
n=X.shape[1] 
# Initial centroids
K=int(input())
Centeroids=np.empty((K, 2))
for i in range(K):
    for j in range(2):
        Centeroids[i][j]=int(input())
print("initial clusters head",Centeroids)

#calculating distances
def dis(Centroids,x):
    dist=((Centroids[0]-x[0])*2)+((Centroids[1]-x[1])*2) 
    return dist
iter=0
while(True):
    iter=iter+1
    print("iteration: ",iter)
    cluster_array=np.array([]).reshape(m,0)
    # cal distance between centroids
    for i in range(m):
        min=dis(Centeroids[0],X[i]) 
        cluster=0 
        for j in range(1,K):
            temp=dis(Centeroids[j],X[i])
            if(temp<min):
                min=temp
                cluster=j
        cluster_array=np.append(cluster_array,cluster)
    cluster_array=cluster_array.astype(int)
    
    #print clusterand find new centroids
    new_centeroids=[[0 for i in range(2)] for j in range(K)]
    for i in range(K):
        print("cluster ",i+1,end=" : ")
        temp_x=0
        temp_y=0
        count=0
        for j in range(m):
            if(cluster_array[j]==i):
                print(X[j],end="  ")
                temp_x=temp_x+X[j][0]
                temp_y=temp_y+X[j][1]
                count=count+1

                new_centeroids[i][0]=temp_x/count
                new_centeroids[i][1]=temp_y/count
        print()
        
    print("new cluster centers: ",new_centeroids)
    print()
    if(np.array_equal(Centeroids,new_centeroids)):
        break
    else:
        centeroids=new_centeroids
    
    
print("final cluster: ",new_centeroids)