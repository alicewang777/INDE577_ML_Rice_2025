# Supervised Learning Algorithms

This directory contains implementations of various supervised learning algorithms:

1. **Perceptron**
2. **Linear Regression**
3. **Logistic Regression**
4. **Neural Networks**
5. **K-Nearest Neighbors**
6. **Decision Trees / Regression Trees**
7. **Random Forests**
8. **Ensemble Methods / Boosting**

This directory collects the abovementioned **eight classic supervised‑learning algorithms** that we hand‑implemented inside the `rice_ml` package and demonstrated on the New York City Uber trip‑records dataset.  
Each sub‑folder contains:

* clean, from‑scratch Python code (no *scikit‑learn*);
* a `README.md`;
* an example notebook showing how to import the model from `rice_ml`, fit it on a task (e.g., predicting high‑fare vs. low‑fare rides), and evaluate performance.

> **Tip:** After cloning the repo, add the project root to your `PYTHONPATH` so you can simply `from rice_ml.perceptron import Perceptron` in any notebook.

## How Supervised Learning Works
Supervised learning involves training a model using labeled data, where the goal is to make predictions or classify data based on patterns learned during training.

1. **Labelled data**  
   We start with pairs `(x_i, y_i)` where `x_i` is a feature vector (pickup longitude, hour of day, …) and `y_i` is the target (fare amount or a boolean “high‑fare” flag).

2. **Train / validation / test split**  
   To gauge generalisation, we hold out a validation set (for model selection) and an independent test set (for the final report).

3. **Model fitting**  
   Each algorithm learns a function `f_θ` that minimises prediction error on the training data—for example, least‑squares loss for regression or cross‑entropy for classification.

4. **Evaluation**  
   We compute metrics exposed in `rice_ml.metrics`:

   - `accuracy` for classification  
   - `mse` / `r2_score` for regression  
   - plus confusion matrices, ROC‑AUC,
     
---

## Challenges & lessons learned

* **From‑scratch math.** Implementing gradient updates, impurity calculations, and back‑prop without libraries surfaced many edge‑cases (division‑by‑zero, numerical overflow).  
* **Imbalanced labels.** Only ~30 % of trips were *high‑fare*; we experimented with class‑weights and stratified sampling.  
* **Computation cost.** KNN’s naïve search scales O($n^2$). We mitigated this with `numpy` vectorisation and down‑sampling.  
* **Over‑ & under‑fitting.** Decision trees required careful pre‑pruning; neural networks needed early‑stopping based on validation loss.  
* **Ensembling trade‑offs.** Boosting improved accuracy but was sensitive to noisy labels; random forests were more robust but less interpretable.

---

## Reproducing our results

1. `git clone https://github.com/alicewang777/INDE577_ML_Rice_2025.git`  
2. `cd INDE577_ML_Rice_2025`  
3. Install Python ≥ 3.10 and run:

   ```bash
   pip install -r requirements.txt   # mostly numpy, pandas, matplotlib
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   ````

4. Open the notebook inside any sub‑folder, run all cells, and compare the printed metrics with those reported in its local `README.md`.

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


