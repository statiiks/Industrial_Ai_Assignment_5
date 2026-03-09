'''
point cloud data is stored as a 2D matrix
each row has 3 values i.e. the x, y, z value for a point

Project has to be submitted to github in the private folder assigned to you
Readme file should have the numerical values as described in each task
Create a folder to store the images as described in the tasks.

Try to create commits and version for each task.

'''
#%%
import matplotlib
import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import NearestNeighbors


#%% utility functions
def show_cloud(points_plt):
    ax = plt.axes(projection='3d')
    ax.scatter(points_plt[:,0], points_plt[:,1], points_plt[:,2], s=0.01)
    plt.show()

def show_scatter(x,y):
    plt.scatter(x, y)
    plt.show()

def get_ground_level(pcd, bins=100, plot=True, save_path=None):
    z_values = pcd[:,2]

    counts, bin_edges = np.histogram(z_values, bins=bins)

    max_bin_idx = np.argmax(counts)

    ground_level = (bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1]) / 2

    if plot:
        plt.figure(figsize=(8, 5))
        plt.hist(z_values, bins=bins, edgecolor="black")
        plt.axvline(ground_level, linestyle="--", label=f"Estimated ground = {ground_level:.3f}")
        plt.xlabel("z value")
        plt.ylabel("Number of points")
        plt.title("Histogram of z values")
        plt.legend()

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    return ground_level

def find_optimal_eps(points, min_samples=5, plot=True, save_path=None):
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(points)
    distances, indices = neighbors_fit.kneighbors(points)

    k_distances = np.sort(distances[:, min_samples - 1])

    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(k_distances)
        plt.xlabel("Points sorted by distance")
        plt.ylabel(f"Distance to {min_samples}th nearest neighbor")
        plt.title("Elbow plot for DBSCAN eps")

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    return k_distances

#%% read file containing point cloud data
dataset_name = "dataset1"
pcd = np.load(f"{dataset_name}.npy")

pcd.shape

#%% show downsampled data in external window
%matplotlib qt
show_cloud(pcd)
#show_cloud(pcd[::10]) # keep every 10th point

#%% remove ground plane

'''
Task 1 (3)
find the best value for the ground level
One way to do it is useing a histogram 
np.histogram

update the function get_ground_level() with your changes

For both the datasets
Report the ground level in the readme file in your github project
Add the histogram plots to your project readme
'''
est_ground_level = get_ground_level(pcd, bins=100, plot=True, save_path=f"hist_{dataset_name}.png")
print("Estimated ground level:", est_ground_level)

ground_pad = 0.2 # ignore points realy close to the ground
pcd_above_ground = pcd[pcd[:,2] > est_ground_level+ ground_pad] 
#%%
print("Original shape:", pcd.shape)
print("Above ground shape:", pcd_above_ground.shape)

#%% side view
show_cloud(pcd_above_ground)


# %%
unoptimal_eps = 10
# find the elbow
clustering = DBSCAN(eps = unoptimal_eps, min_samples=5).fit(pcd_above_ground)

#%%
clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, clusters)]

# %%
# Plotting resulting clusters
plt.figure(figsize=(10,10))
plt.scatter(pcd_above_ground[:,0], 
            pcd_above_ground[:,1],
            c=clustering.labels_,
            cmap=matplotlib.colors.ListedColormap(colors),
            s=2)


plt.title('DBSCAN: %d clusters' % clusters,fontsize=20)
plt.xlabel('x axis',fontsize=14)
plt.ylabel('y axis',fontsize=14)
plt.show()


#%%
'''
Task 2 (+1)

Find an optimized value for eps.
Plot the elbow and extract the optimal value from the plot
Apply DBSCAN again with the new eps value and confirm visually that clusters are proper

https://www.analyticsvidhya.com/blog/2020/09/how-dbscan-clustering-works/
https://machinelearningknowledge.ai/tutorial-for-dbscan-clustering-in-python-sklearn/

For both the datasets
Report the optimal value of eps in the Readme to your github project
Add the elbow plots to your github project Readme
Add the cluster plots to your github project Readme
'''

k_distances = find_optimal_eps(
    pcd_above_ground,
    min_samples=5,
    plot=True,
    save_path=f"elbow_{dataset_name}.png"
)

#Zoom on the knick
plt.figure(figsize=(8, 5))
plt.plot(k_distances)
plt.xlabel("Points sorted by distance")
plt.ylabel("Distance to 5th nearest neighbor")
plt.title("Elbow plot for DBSCAN eps (zoomed)")
plt.ylim(0, 3)
plt.xlim(len(k_distances) * 0.9, len(k_distances))
plt.grid(True)
plt.show()
plt.savefig(f"elbow_zoomed_{dataset_name}.png", dpi=300, bbox_inches="tight")

#%%
optimal_eps = 2.0 # From Plot + viszual approved
clustering = DBSCAN(eps=optimal_eps, min_samples=5).fit(pcd_above_ground)

#%%
clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, clusters)]

#%%
# Plotting resulting clusters
plt.figure(figsize=(10,10))
plt.scatter(pcd_above_ground[:,0], 
            pcd_above_ground[:,1],
            c=clustering.labels_,
            cmap=matplotlib.colors.ListedColormap(colors),
            s=2)


plt.title('DBSCAN: %d clusters' % clusters,fontsize=20)
plt.xlabel('x axis',fontsize=14)
plt.ylabel('y axis',fontsize=14)
plt.show()
plt.savefig(f"Clusterd_{dataset_name}.png", dpi=300, bbox_inches="tight")

#%%
'''
Task 3 (+1)

Find the largest cluster, since that should be the catenary, 
beware of the noise cluster.

Use the x,y span for the clusters to find the largest cluster

For both the datasets
Report min(x), min(y), max(x), max(y) for the catenary cluster in the Readme of your github project
Add the plot of the catenary cluster to the readme

'''

# %%
glabels = clustering.labels_
unique_lables = [label for label in set(labels) if label != -1]

cluster_info = []

for label in unique_lables:
    cluster_points = pcd_above_ground[labels == label]

    min_x = cluster_points[:, 0].min()
    max_x = cluster_points[:, 0].max()
    min_y = cluster_points[:, 1].min()
    max_y = cluster_points[:, 1].max()

    x_span = max_x - min_x
    y_span = max_y - min_y
    span_score = x_span + y_span

    cluster_info.append({
        "label": label,
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "x_span": x_span,
        "y_span": y_span,
        "span_score": span_score
    })
# %%
largest_cluster = max(cluster_info, key=lambda c: c["span_score"])

print("Selected cluster label:", largest_cluster["label"])
print("min_x:", largest_cluster["min_x"])
print("max_x:", largest_cluster["max_x"])
print("min_y:", largest_cluster["min_y"])
print("max_y:", largest_cluster["max_y"])
print("x_span:", largest_cluster["x_span"])
print("y_span:", largest_cluster["y_span"])
# %%
catenary_label = largest_cluster["label"]
catenary_points = pcd_above_ground[labels == catenary_label]

plt.figure(figsize=(8, 8))
plt.scatter(catenary_points[:, 0], catenary_points[:, 1], s=2)
plt.title(f"Catenary cluster (label={catenary_label})")
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.savefig(f"catenary_{dataset_name}.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
