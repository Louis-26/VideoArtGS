# tasks
- ✅draw pipeline(PPT/drawio) procedure graph, from part number, position, x number, from method
- ✅side by side analyze the error, qualitatively take a look at the videos, visualize the axis
- ⌛use part segmentation ground truth at first, adjust network structure, use LoRA update to finetune the model (15 for train, 5 for test)


# work finished
- finish the pipeline procedure graph for both original VideoArtGS and refined version by Figma, with link: https://www.figma.com/design/7dDTR57ZKdyMfOiuJXp0s8/VideoArtGS-PAT-procedure-graph?node-id=2-5&t=X7Q5HsEE1fvP6ysV-1
- main reason for failure is due to the incorrect part segmentation, either from missing or incorrectly classified fraction for certain parts 
- use `100481  101284  101287  101808  101908  103015  103811  10489  10655  1280  168  25493  30666  31249  45194` as training scenes to finetune the model, and `45503  45612  47648  8961  9016` as test scenes.

# reference
https://sites.google.com/view/reartgs2/home/

