# Tasks
Fully train the PAT model with initialized weights from the pretrained model 


1. train seen objects(utilize 70% **moving** frames as training set and 30% **moving** frames as test set for all objects)

- same object, with different views for training and test dataset

2. train unseen objects(utilize all frames from 15 objects as training set and 5 objects as test set)

- different objects from the training dataset and test dataset



Evaluation:

- for seen and trained objects
    - compare CD loss

- for unseen objects(for generalization)
    - check performance
    - finetune model (RGB/depth)

PAT evaluation:
- `Part segmentation mIoU`
- `Joint type accuracy`
- `Axis angular error`
- `Origin / axis-line error`
- `Part-center error`


VideoArtGS + PAT:
- `CD loss`
- `joint state error`
- `axis error`
- `position error`


# finished tasks

- ✅finish full PAT training with initialized weights from the pretrained model with two groups
	- group 1: train scenes: `101287  101808  101908  103015  10489
							  10655   1280    168     25493   30666
							  31249   45503   45612   8961    9016`
				test scenes: `100481  101284  103811  45194  47648`
	- group 2: first 70% of moving frames for training, last 30% of moving frames for testing for all objects
- ✅finish PAT checkpoint saving
	- [group 1](../particulate/model_ckpt/PAT_5/b1/pat_5b1.pt)
	- [group 2](../particulate/model_ckpt/PAT_5/b2/pat_5b2.pt)
- ✅finish evaluation for both experiments **given** the trained PAT checkpoint

- ✅list the comparison for [group 1](../experiment_results/PAT_5b1/videoartgs_sapien_results.txt) and [group 2](../experiment_results/PAT_5b2/videoartgs_sapien_results.txt)




  

  
# Next steps

1. collect and compare the baselines 

- Videoartgs baselines reproduction(qualitative, quantitative)
  - articulateanything
  - RSRD
  - video2articulation

- Videoartgs+PAT(already finished✅)

2. Methodology + Experiment in paper


3. Ablation study

	- vs videoartgs
	- transformer architecture(without certain features, what will be the performance)



