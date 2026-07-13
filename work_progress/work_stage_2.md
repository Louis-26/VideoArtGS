# Date
06/21/2026 - 07/9/2026

# tasks
- ✅read through `Particulate` paper and reproduce code results 
- ✅combine the idea from VideoArtGS and Particulate, replace the tedious postprocessing processes with a learnable transformer
- ✅code implementation for training/testing and qualitative/quantitative evaluation
- ⌛figure out limitation/challenge/idea motivation/solution
- ⌛idea/implementation pipeline
- ✅baseline comparison
- ✅ablation study design

# finished work summary
- Go over the `VideoArtGS` pipeline and step into engineering details, with updated [README file](../README.md) for a clear illustration for the work procedure
- Integrate overall particulate pretrained model and relevant scripts into the VideoArtGS pipeline, mapping from 3D Gaussian primitives from 3DGS results of canonical gaussians to complex articulation parameters 
- Finish [init_deform_PAT.py](../init_deform_PAT.py) implementation and testing, deriving qualitative and quantitative results for the new setup
- Finish [idea_analysis.md](../overview/idea_analysis.md) for the limitation/challenge/idea motivation/solution
- Finish VideoArtGS pipeline with PAT integration [VideoArtGS+PAT_pipeline](../overview/VideoArtGS+PAT_pipeline.md)
- Quantitative results have been compared between the paper's and the PAT model integration in [paper_reproduce.md](../experiment_results/paper_reproduce.md)
- attempt to test the effect of the following factors
    - 3D coordinates
    - segmentation feature





# potential next steps
Finetune the PAT pipeline, possibly in
- input dimension/preprocessing
- model architecture


# reference
⭐[Particulate: Feed-Forward 3D Object Articulation](https://ruiningli.com/particulate)