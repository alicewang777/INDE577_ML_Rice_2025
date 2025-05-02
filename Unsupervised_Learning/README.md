# Image Compression using SVD (from `rice_ml` Package)

## Overview

This project demonstrates how to compress grayscale images using the Singular Value Decomposition (SVD) algorithm. The implementation is based on a custom machine learning package named `rice_ml`, which was developed as part of a broader toolkit for supervised and unsupervised learning tasks.

The compression technique is applied to an image of a cat for visual demonstration, while the same `rice_ml` package has also been used for analyzing real-world datasets.

## Algorithm Description

I use the mathematical formulation of SVD to decompose a 2D grayscale image matrix `M` into three components:

M = U · Σ · Vᵗ


- `U`: An orthogonal matrix containing the left singular vectors.
- `Σ`: A diagonal matrix of singular values (in descending order).
- `Vᵗ`: An orthogonal matrix containing the right singular vectors.

By retaining only the top \( k \) singular values and their corresponding vectors (a rank-\( k \) approximation), I reconstruct a compressed version of the image. This method enables a trade-off between compression and image quality.

## Visual Explanation

The following results illustrate how varying the number of retained singular values \( k \) impacts image quality:

![image_compression_with_the_singular_value_decomposition_2](https://github.com/user-attachments/assets/9388856a-6222-49f5-9166-b0904cf92b05)

At low \( k \) (e.g., 5 or 20), the image is blurry and lacks detail. As \( k \) increases, the image becomes sharper. At \( k = 200 \), the compressed image is visually nearly identical to the original.

## SVD Process Illustration

The figure below illustrates the geometric interpretation of Singular Value Decomposition (SVD). Any linear transformation matrix `M` can be decomposed into:

1. A **rotation** via the matrix `Vᵗ` (right singular vectors)
2. A **scaling** along orthogonal axes via the diagonal matrix `Σ` (singular values)
3. Another **rotation** via the matrix `U` (left singular vectors)

This means the transformation `M` is equivalent to rotating a vector space, stretching it along new orthogonal directions, and then rotating again.

![image_compression_with_the_singular_value_decomposition_3](https://github.com/user-attachments/assets/ac6789df-a94d-4db3-98a1-ae204ee54f61)

## Dataset

- **Image**: A single RGBA cat image, converted to grayscale and resized to 256x256 for computational efficiency.

## Instructions for Reproducing Results

1. Load the image and convert it to grayscale.
2. Use the `compress_svd(image, k)` function from the `rice_ml` package:
   ```python
   from rice_ml.svd import compress_svd
   compressed = compress_svd(img_gray, k=50)
   ```
3. Visualize the compressed output using `matplotlib`.

> Note: This implementation uses NumPy’s `np.linalg.svd` and does not rely on external ML libraries like scikit-learn.

## Reference

Miczek, D. (2023, December 17). SVD Image Compression, Explained. Retrieved from https://dmicz.github.io/machine-learning/svd-image-compression/
