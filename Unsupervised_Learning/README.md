# Unsupervised Learning Algorithms

This directory documents four classical unsupervised‑learning techniques implemented with the `rice_ml` package.  
The first three notebooks analyse a 5 000‑row sample of the public Uber pick‑up dataset; the fourth compresses a high‑resolution cat photograph.
All code, diagnostics and figures are reproduced in the accompanying Colab Notebooks.

## What Is Unsupervised Learning?  
In supervised learning we train a model with **labelled examples**—each observation comes with a ground‑truth target *y*.  
**Unsupervised learning removes those labels** and asks algorithms to *let the data speak for itself* by:

1. **Grouping** similar observations together (clustering).  
2. **Compressing / projecting** high‑dimensional data onto a lower‑dimensional manifold (dimensionality reduction).  
3. **Detecting anomalies** that deviate from the dominant structure (outlier detection, density estimation).  

Because there is no external “answer key,” the objective is usually framed as *optimising an internal metric* (e.g., minimising within‑cluster variance in K‑Means, maximising density connectivity in DBSCAN, or maximising explained variance in PCA).  
The resulting patterns often serve as **exploratory insights** or **pre‑processing steps** that feed into downstream supervised tasks.

---

## 1 K‑Means Clustering

<img width="584" alt="kmeans_clustering_1" src="https://github.com/user-attachments/assets/c33fb8e8-73c2-4d1b-a94c-d7657ffbcb4f" />

`kmeans_clustering.ipynb` applies a hand‑written K‑Means routine to a nine‑dimensional feature set that mixes spatial coordinates with temporal and behavioural variables (hour, day‑of‑week, passenger count, Manhattan distance, and an hour × passenger interaction). The notebook shows the full training script, an inertia / silhouette study for choosing *k*, and both static and interactive map visualisations of the resulting four‑cluster solution.
The clusters separate airport transfers, office‑hour commutes, late‑night leisure rides, and low‑density suburban trips, illustrating how simple Euclidean partitioning can uncover interpretable market segments.

![kmeans_clustering_2](https://github.com/user-attachments/assets/1468a9eb-bf24-4c35-bf65-6fea8b9ed11b)

![kmeans_clustering_3](https://github.com/user-attachments/assets/930353cf-b746-439d-8e0d-c3006b2642c8)

---

## 2 DBSCAN

<img width="630" alt="dbscan_1" src="https://github.com/user-attachments/assets/ce9eea2d-2402-48d2-8969-c92f1660164f" />

`dbscan.ipynb` demonstrates Density‑Based Spatial Clustering of Applications with Noise. Because DBSCAN does not require a pre‑specified number of clusters, it is well‑suited to the heterogeneous density of urban mobility data. After z‑scaling the same nine features, the notebook tunes `eps` via the k‑distance heuristic and sets `min_samples = 10`. The outcome reveals organically shaped neighbourhood clusters while flagging infrequent or erroneous pick‑ups as noise.

![dbscan_2](https://github.com/user-attachments/assets/34231b8c-f316-4779-b850-f19a9af90360)

![dbscan_3](https://github.com/user-attachments/assets/83dfbb04-a127-4ddd-9db7-37827c25e4c4)

---

## 3 Principal Component Analysis

<img width="1073" alt="principal_component_analysis_1" src="https://github.com/user-attachments/assets/98bd7c84-6915-4e7a-b210-e202471dc1f4" />

`principal_component_analysis.ipynb` reduces the Uber feature matrix to two orthogonal principal components that preserve about 70 % of total variance.
The projection compresses information on geography, timing and ride length into a plane where latent usage patterns become visually distinct. Overlaying the earlier K‑Means labels confirms the consistency of structure across methods and simplifies downstream dashboarding.

![principal_component_analysis_2](https://github.com/user-attachments/assets/bb14b9fe-a92a-417a-9012-a60e5e9c851f)

---

## 4 Image Compression via Singular‑Value Decomposition

![image_compression_with_the_singular_value_decomposition_3](https://github.com/user-attachments/assets/6431da29-ce61-4f83-905e-c2bf5960be57)

`image_compression_with_the_singular_value_decomposition.ipynb` loads a 512 × 512 RGB cat image, converts it to grayscale, and reconstructs it with rank‑*k* SVD approximations for *k* = 5, 20, 50, 100, 200.
The notebook visualises the trade‑off between compression ratio and perceptual quality: with *k* = 100 the file size falls by roughly 6× while structural fidelity remains high.

![image_compression_with_the_singular_value_decomposition_2](https://github.com/user-attachments/assets/104342ca-bbe6-4521-9b83-1a7c06846340)

---

## Challenges and Practical Notes

* **Feature scaling and parameter selection** K‑Means requires a well‑chosen *k*; DBSCAN is sensitive to `eps`. Both notebooks therefore include systematic heuristics (elbow, silhouette, k‑distance) instead of arbitrary defaults.  
* **Heterogeneous density** In mixed urban data, centroid‑based methods may split dense downtown traffic but smear sparsely populated zones. DBSCAN mitigates this by modelling density directly, yet demands careful tuning to avoid over‑fragmentation.  
* **Dimensionality versus interpretability** PCA confirms that most behavioural variance lies on two axes, justifying 2‑D visualisations, but interpreting rotated components still needs inspection of the loading matrix.  
* **SVD rank choice** Image quality improves rapidly up to about *k* = 100; beyond that, gains are marginal.

---

## Reproduction

```bash
git clone https://github.com/alicewang777/INDE577_ML_Rice_2025.git
cd INDE577_ML_Rice_2025/Unsupervised_Learning
pip install -e ../rice_ml          # editable install of custom package
# open the desired *.ipynb file in Colab or Jupyter and run all cells
```
---

## References

Miczek, D. (2023, December 17). SVD Image Compression, Explained. Retrieved from https://dmicz.github.io/machine-learning/svd-image-compression/

Im, J. (2018, December 6). Introduction to PCA (Principal Component Analysis). Medium. https://medium.com/@jamesim2077/introduction-to-pca-principal-component-analysis-c26dffe2a857​

Omarzai, F. (2024, July 20). DBSCAN In Depth. Medium. https://medium.com/@fraidoonomarzai99/dbscan-in-depth-3fa4a8dbd3af​

Patel, A. (2021, June 17). K-Means Clustering. Medium. https://imakash3011.medium.com/k-means-clustering-ef8e9258d76a​



