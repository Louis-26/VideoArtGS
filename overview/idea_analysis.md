# limitation
I notice that quite a number of scenes from VideoArtGS sapien dataset have significant failure cases, where the axis error mean and variance are much larger than the baseline results from the original paper(for instance, 100481, 101284, 101287, ...) 


# challenge
Part Articulation Transformer (PAT) was trained with a different training dataset from the VideoArtGS dataset, and it was trained on mesh data, which has a different data architecture from the gaussians.


# idea motivation
PAT can provide information to predict a mapping from the time to articulation parameters, which can be further used to initialize the deformation field for the VideoArtGS pipeline. 


# solution
- fine tune pretrained PAT models on VideoArtGS dataset, and use refined model for inference again for deformation field initialization
- adjust the PAT model architecture to process gaussian primitives(number of layers, activation function, number of heads)
- adjust input features and corresponding dimension(include 3D coordinates, segmentation feature, )