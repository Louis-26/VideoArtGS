# Evaluation Metric

## Articulation Estimation
It records the mean error plusorminus one standard deviation between the predicted and ground-truth articulation parameters

- Revolute Joint Estimation:
    - Axis Error  
    - Position Error
    - State Error
- Prismatic Joint Estimation:
    - Axis Error
    - State Error


## Mesh Reconstruction
- CD-w (Chamfer Distance - whole)
- CD-m (Chamfer Distance - movable)
- CD-s (Chamfer Distance - static)

# Reference
## Chamfer Distance
Given two point clouds $P_1=\{x_i \in R^3\}_{i=1}^{n}$ and $P_2=\{y_j \in R^3\}_{j=1}^{m}$, the Chamfer Distance is defined as:
$$
CD(P_1, P_2) = \frac{1}{2n}\sum_{i=1}^{n} \min_{y_j \in P_2} \|x_i - y_j\|^2 + \frac{1}{2m}\sum_{j=1}^{m} \min_{x_i \in P_1} \|y_j - x_i\|^2
$$