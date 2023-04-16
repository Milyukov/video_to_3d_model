# video_to_3d_model
A project dedicated to 3D reconstruction from video sequence

Camera model and parametrization
=
Extrinsics
-
1. $ \begin{pmatrix*}[l] t_x & t_y & t_z \end{pmatrix*}^T = \bold \it t_c $ - camera center coordinates in world.

2. $ \begin{pmatrix*}[l] \theta_{roll} & \theta_{pitch} & \theta_{yaw} \end{pmatrix*}^T$ - Euler rotation angles (extrinsic or intrinsic) representing camera orientation in world. Extrinsic rotations are rotation around fixed world axes. Intrinsic rotations are rotations around camera axes.

World coordinate axes and rotations are **right-handed**. Camera orientation is set up by rotation matrix $ \bold R$ with rotation sequence roll (z) -> pitch (x) -> yaw (y):
$$   \bold R^{ext}(\theta_{roll},\theta_{pitch}, \theta_{yaw})  = \bold R_{yaw}(\theta_{yaw}) \cdot \bold R_{pitch}(\theta_{pitch}) \cdot \bold R_{roll}(\theta_{roll})  $$
$$   \bold R^{int}(\theta_{roll},\theta_{pitch}, \theta_{yaw})  = \bold R_{roll}(\theta_{roll}) \cdot \bold R_{pitch}(\theta_{pitch}) \cdot \bold R_{yaw}(\theta_{yaw})  $$



Respective transformation matrix $ \bold M_{w2c} $ for changing coordinate system from world axes to camera axes is given as:
$$ \bold M_{w2c} = \bold R^T(\theta_{roll},\theta_{pitch}, \theta_{yaw}) $$

In homogeneous form point $\bold \it p_w = \begin{pmatrix} x_w & y_w & z_w & 1 \end{pmatrix}^T $ in world axes transforms to camera axes as point $\bold \it p_c = \begin{pmatrix} x_c & y_c & z_c & 1 \end{pmatrix}^T $:
$$ \begin{pmatrix} x_c \\ y_c \\ z_c \\ 1\end{pmatrix} = \begin{pmatrix} \bold M_{w2c} & \bold - M_{w2c} \cdot \bold \it t_c \\ 0_{1 \times 3} & 1 \end{pmatrix} \cdot  \begin{pmatrix} x_w\\ y_w\\ z_w\\ 1 \end{pmatrix}$$

Intrinsics
-

Projection onto camera matrix
-