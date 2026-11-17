# overall task
Given monocular multiview video frames, reconstruct articulated 3D object digital twin.
Given specific scene with **T** input frames, including **M** canonical static multiview video frames and **T-M** deformed multiview frames(given camera poses), estimate the following segmentation, articulation and motion parameters:
- segmentation
    - part radial extent 
    - part number $K$
    - part center $c_k$, with $k \in \{1, \ldots, K\}$
- articulation
    - joint type $\tau_k$, with $k \in \{2, \ldots, K\}$
    - prismatic
        - articulation axis $d_k$, with $k \in \{2, \ldots, K\}$
    - revolute
        - articulation axis $d_k$, with $k \in \{2, \ldots, K\}$
        - articulation origin $o_k$, with $k \in \{2, \ldots, K\}$
- motion
    - joint motion limits $\theta_k^{min}, \theta_k^{max}$, with $k \in \{2, \ldots, K\}$
    - joint state $\theta_k$, with $k \in \{2, \ldots, K\}$

After that, reconstruct 4D object digital twin with gif/mp4 and mesh visualization.

Finally, we can comprehensively analyze the 3D object with arbitrary perspective and arbitrary time. 