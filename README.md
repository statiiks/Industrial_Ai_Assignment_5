# Assignment 5 - LiDAR Point Cloud Analysis
This project contains the solution for Assignment 5 in Industrial AI. 
The work was first developed on `dataset1.npy` and then tested on `dataset2.npy`.

## Task 1 - Ground level estimation
The ground level was estimated using a histogram of the z values. 
The strongest peak in the histogramm was used as the estimated ground level.

### Dataset 1
- Estimated ground level: **61.250**
![Histogram dataset1](hist_dataset1.png)

### Dataset 2
- Estimated ground level: **61.265**
![Histogram dataset2](hist_dataset2.png)

---

## Task 2 - DBSCAN eps optimization

The eps value for DBSCAN was estimated using an elbow plot based on the distance to the 5th nearest neighbor.  
The selected eps value was then validated visually with the resulting cluster plot.

### Dataset 1
- Selected eps: **2.0**

![Elbow dataset1](elbow_dataset1.png)

![Cluster plot dataset1](Clusterd_dataset1.png)

### Dataset 2
- Selected eps: **2.0**

![Elbow dataset2](elbow_dataset2.png)

![Cluster plot dataset2](Clusterd_dataset2.png)

---

## Task 3 - Catenary cluster extraction

The catenary was identified as the largest meaningful cluster based on the x/y span.  
The noise cluster with label `-1` was ignored.

### Dataset 1
- Selected cluster label: **1**
- min(x): **26.498**
- max(x): **62.140**
- min(y): **80.019**
- max(y): **159.960**
- x span: **35.642**
- y span: **79.941**

![Catenary dataset1](catenary_dataset1.png)

### Dataset 2
- Selected cluster label: **5**
- min(x): **10.179**
- max(x): **37.007**
- min(y): **0.043**
- max(y): **79.976**
- x span: **26.828**
- y span: **79.933**

![Catenary dataset2](catenary_dataset2.png)
