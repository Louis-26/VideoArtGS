# overall task
Given monocular multiview video frames, reconstruct articulated 3D object digital twin.
Given specific scene with **T** input frames, including **M** canonical static multiview video frames and **T-M** deformed multiview frames(given camera poses), estimate the following segmentation, articulation and motion parameters:
- segmentation
    - Part Radial Extent 
    - per-Gaussian part id
    - part number
    - part center
- articulation
    - joint type
    - prismatic
        - articulation axis
    - revolute
        - articulation axis
        - articulation origin
- motion
    - joint motion limits
    - joint state(degree number or displacement number)

After that, reconstruct 4D object digital twin with gif/mp4 and mesh visualization.

Finally, we can comprehensively analyze the 3D object with arbitrary perspective and arbitrary time. 