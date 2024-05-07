import numpy as np
from scipy.spatial.transform import Rotation as R

# Current orientation quaternion
quad_quat = np.array([1, 0, 0, 0])

# Desired acceleration vector (normalized)
des_accn = np.array([0, 1, 9.8])

# Define the desired yaw angle (assuming it's 0 for hovering)
yaw_sp = 0.0

# Calculate full desired quaternion qd_full
body_z = -des_accn / np.linalg.norm(des_accn)  # In ENU frame
y_C = np.array([-np.sin(yaw_sp), np.cos(yaw_sp), 0.0])
body_x = np.cross(y_C, body_z)
body_x = body_x / np.linalg.norm(body_x)
body_y = np.cross(body_z, body_x)
R_sp = np.array([body_x, body_y, body_z]).T
Rot = R.from_matrix(R_sp)
quat = Rot.as_quat()
qd_full = np.array([quat[3], quat[0], quat[1], quat[2]])

# Calculate quaternion error
e_z = quad_quat[1:4]
qe_red = np.zeros(4)
qe_red[0] = np.dot(e_z, des_accn) + np.sqrt((1 - np.dot(e_z, e_z)) * (1 - np.dot(des_accn, des_accn)))
qe_red[1:4] = np.cross(e_z, des_accn)
qe_red = qe_red / np.linalg.norm(qe_red)

# Calculate reduced desired quaternion
qd_red = qe_red * quad_quat

# Calculate mixed desired quaternion
q_mix = qd_full * np.sign(qd_full[0])

# Calculate resulting desired quaternion
yaw_w = 1.0  # Yaw control weight
qd = qd_red * np.array([np.cos(yaw_w * np.arccos(q_mix[0])), 0, 0, np.sin(yaw_w * np.arcsin(q_mix[3]))])

print("Resulting desired quaternion (qd):", qd)