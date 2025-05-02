![image](https://github.com/user-attachments/assets/4fbaac79-fc1b-4bab-a2f0-5dba45879371)

In busy New York City, Uber trips naturally gather in clear hotspots. The kernel density map above plots pickup and drop‑off points on an OpenStreetMap background. Bright reds and yellows mark areas with heavy demand such as Midtown and Lower Manhattan, while green and blue dots show quieter zones.

This first look at the data was my starting point. It reminded me that ride patterns change over time and differ by neighborhood. Guided by those insights, the project rebuilds several classic machine‑learning models from scratch—linear regression, random forests, K‑Means, DBSCAN, and others. A reusable pipeline handles data cleaning, feature engineering, and visualization so I can compare each model on prediction, classification, and clustering tasks. The goal is to show how hands‑on data science can help us understand and improve real‑world travel behavior.


# INDE577_ML_Rice_2025  
_Final Project for INDE 577: Data Science & Machine Learning_

**Author:** Alice Wang

**Course:** INDE 577 / CMOR 438 – Rice University

**Instructor:** Prof. Randy Davila

**Term:** Spring 2025

---

## Project Overview
This repository is the deliverable for my INDE 577 final project.  It showcases **from‑scratch implementations** of core machine‑learning algorithms, organized around two learning paradigms:

| Paradigm | Directory | Algorithms Implemented | Notebook Examples |
|----------|-----------|------------------------|-------------------|
| **Supervised Learning** | `Supervised_Learning` | Perceptron • Linear & Logistic Regression • Neural Networks • K‑Nearest Neighbors • Decision / Regression Trees • Random Forests • Boosting | End‑to‑end Uber‑trip prediction & classification notebooks for each model |
| **Unsupervised Learning** | `Unsupervised_Learning` | K‑Means • DBSCAN • Principal Component Analysis • SVD‑based Image Compression | Uber GPS clustering, PCA on high‑dimensional features, and cat‑image compression demo |

A lightweight **Python package**, **`rice_ml`**, wraps every algorithm in a reusable API so the exact same code can be imported in any notebook or downstream project.

---



## Repository Structure
```

INDE577_ML_Rice_2025/
├── Supervised_Learning/        # Notebooks + README for each supervised model
├── Unsupervised_Learning/      # Notebooks + README for each unsupervised model
├── rice_ml/                    # Reusable package (pip‑installable locally)
│   ├── <algorithm>.py           # e.g., perceptron, decision_tree, ...
│   ├── metrics.py              # accuracy, MSE, R², etc.
│   └── __init__.py
└── README.md                   # ← you are here


````

### Why a Separate Package?
* **Reusability** – Import models anywhere via `import rice_ml.<model>`.
* **Unit Testing** – Algorithms are decoupled from notebooks, making pytest coverage simple.
* **Packaging Practice** – Mirrors real‑world library structure (init file, clear APIs, docstrings).

---

## Datasets Used
| Dataset | Purpose | Source |
|---------|---------|--------|
| **NYC Uber Trips** | Regression & classification tasks (fare prediction, demand hotspots) | TLC public data on Kaggle |
| **Cat Image (PNG)** | Demonstrates lossy image compression with SVD | Public‑domain sample image |

*All datasets are loaded or downloaded automatically inside the notebooks; no manual steps necessary.*

---

## Quick Start
1. **Clone** the repo  
   ```bash
   git clone https://github.com/alicewang777/INDE577_ML_Rice_2025.git
   cd INDE577_ML_Rice_2025
   ````

2. **Set up** a Python environment (recommended via `conda` or `venv`)

   ```bash
   pip install -r requirements.txt   # only numpy, pandas, matplotlib, jupyter
   ```

3. **Run notebooks**

   ```bash
   jupyter notebook
   # open any .ipynb under Supervised_Learning/ or Unsupervised_Learning/
   ```

4. **Use the package elsewhere**

   ```python
   from rice_ml.linear_regression import LinearRegression
   model = LinearRegression()
   model.fit(X_train, y_train)
   ```
---

## Results & Visualisations

This root README showcases a NYC Uber heat‑map from the EDA stage, while each algorithm‑specific README and notebook include additional visualizations such as confusion matrices.

---

## References

Neurallearner. (2017, October 25). Deep learning: An extension of the perceptron. Steemit. https://steemit.com/technology/@neurallearner/deep-learning-an-extension-of-the-perceptron

Knoldus Inc. (2018, March 28). MachineX: Simplifying Logistic Regression. Medium. https://medium.com/knoldus/machinex-simplifying-logistic-regression-93b2e6d88a8a

Bonnet, A. (2023, November 24). What is Ensemble Learning? Encord. https://encord.com/blog/what-is-ensemble-learning/​

Yehoshua, R. (2023, March 25). Random Forests. Medium. https://medium.com/@roiyeho/random-forests-98892261dc49

Parihar, G. (2020, June 3). Machine Learning: Decision Tree Regression. Medium. https://medium.com/analytics-vidhya/machine-learning-decision-tree-regression-ff8563ffaf52​

Sachinsoni. (2023, June 11). K Nearest Neighbours — Introduction to Machine Learning Algorithms. Medium. https://medium.com/@sachinsoni600517/k-nearest-neighbours-introduction-to-machine-learning-algorithms-9dbc9d9fb3b2​

Ball, P. (2014, July 8). ‘Wisdom of the crowd’: The myths and realities. BBC Future. https://www.bbc.com/future/article/20140708-when-crowd-wisdom-goes-wrong​

BBC. (2011, January 4). The Code - The Wisdom of the Crowd [Video]. YouTube. https://www.youtube.com/watch?v=iOucwX7Z1HU

Pramoditha, R. (2022, February 1). Overview of a Neural Network’s Learning Process. Medium. https://medium.com/data-science-365/overview-of-a-neural-networks-learning-process-61690a502fa

Miczek, D. (2023, December 17). SVD Image Compression, Explained. Retrieved from https://dmicz.github.io/machine-learning/svd-image-compression/

Im, J. (2018, December 6). Introduction to PCA (Principal Component Analysis). Medium. https://medium.com/@jamesim2077/introduction-to-pca-principal-component-analysis-c26dffe2a857​

Omarzai, F. (2024, July 20). DBSCAN In Depth. Medium. https://medium.com/@fraidoonomarzai99/dbscan-in-depth-3fa4a8dbd3af​

Patel, A. (2021, June 17). K-Means Clustering. Medium. https://imakash3011.medium.com/k-means-clustering-ef8e9258d76a​


