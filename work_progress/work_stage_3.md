# tasks
- ✅draw pipeline(PPT/drawio) procedure graph, from part number, position, x number, from method
- ✅side by side analyze the error, qualitatively take a look at the videos, visualize the axis
- ✅use part segmentation ground truth at first, adjust network structure, use LoRA update to finetune the model (15 for train, 5 for test)


# work finished
- finish the pipeline procedure graph for both original VideoArtGS and refined version by Figma, with link: https://www.figma.com/design/7dDTR57ZKdyMfOiuJXp0s8/VideoArtGS-PAT-procedure-graph?node-id=2-5&t=X7Q5HsEE1fvP6ysV-1
- figure out failure cases, (main reason failure is due to the incorrect part segmentation, either from missing or incorrectly classified fraction for certain parts)
- switch original pretrained model checkpoint to the correct one, include part features from PartField extraction, significantly improving the performance
- use rest as training scenes to finetune the PAT, and `100481, 101284，103811, 45194, 47648` as test scenes
- refine execution pipeline, add arguments to each script for more flexible usage and derive the overall execution script
- complete the overall dataset preprocessing setup and finish verification
- complete detailed summary of both videoartgs and videoartgs+pat methodology

# results
With model switching, the results are much better than the first edition(after switching the pretrained model, refine PAT features and include partfiled features during injection procedure), but still not really satisfactory. The main failure cases include(visible [here](../experiment_results/PAT_1/videoartgs_sapien_results.txt))
- 100481
- 101284
- 103811
- 47648
- 45194
failure reason is similar: part segmentation is not accurate.
After the finetuning, the results turned out to be even worse than the pretrained version.

# discovery
## limitation
I notice that quite a number of scenes from VideoArtGS sapien dataset have significant failure cases, where the axis error mean and variance are much larger than the baseline results from the original paper(for instance, 100481, 101284, 101287, ...) 


## challenge
Part Articulation Transformer (PAT) was trained with a different training dataset from the VideoArtGS dataset, and it was trained on mesh data, which has a different data architecture from the gaussians.


## idea motivation
PAT can provide information to predict a mapping from the time to articulation parameters, which can be further used to initialize the deformation field for the VideoArtGS pipeline. 


## solution
- fine tune pretrained PAT models on VideoArtGS dataset, and use refined model for inference again for deformation field initialization
- adjust the PAT model architecture to process gaussian primitives(number of layers, activation function, number of heads)
- adjust input features and corresponding dimension(include 3D coordinates, segmentation feature, )

# next step
Hyperparameter tuning might be helpful, where the hyperparameters include:
- lora rank
- attention block number
- input feature dimension

# reference
https://sites.google.com/view/reartgs2/home/

