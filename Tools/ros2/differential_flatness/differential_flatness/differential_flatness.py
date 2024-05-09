import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from ardupilot_msgs.msg import AngularVelandAccn
from geometry_msgs.msg import Quaternion, Vector3, PoseStamped, TwistStamped
from .trajectory import Trajectory
import numpy as np
from numpy.linalg import norm
import math
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.executors import MultiThreadedExecutor
# import scipy.spatial.transform as sci
from scipy.spatial.transform import Rotation as R

class IMUPublisherSubscriber(Node):
    def __init__(self):
        super().__init__('imu_publisher_subscriber')
        self.publisher = self.create_publisher(AngularVelandAccn, '/ap/cmd_ang_goals', 10)

        self.pose_subscriber_ = self.create_subscription(
            PoseStamped, '/ap/pose/filtered', self.pose_callback, 10)

        self.twist_subscriber_ = self.create_subscription(
            TwistStamped, '/ap/twist/filtered', self.velocity_callback, 10)
        
        self.timer = self.create_timer(0.01,self.prepare_angular_vel_accn_message)
        self.start_time = self.get_clock().now().nanoseconds
        self.desPos = np.zeros(3)    # Desired position (x, y, z)
        self.desVel = np.zeros(3)    # Desired velocity (xdot, ydot, zdot)
        self.desAcc = np.zeros(3)    # Desired acceleration (xdotdot, ydotdot, zdotdot)
        self.desThr = np.zeros(3)    # Desired thrust in N-E-D directions (or E-N-U, if selected)
        self.desEul = np.zeros(3)    # Desired orientation in the world frame (phi, theta, psi)
        self.desPQR = np.zeros(3)    # Desired angular velocity in the body frame (p, q, r)
        self.yawFF = 0.0         # Desired yaw speed
        self.desJerk = np.zeros(3)   # Desired Jerk (xdotdotdot, ydotdotdot, zdotdotdot)
        self.desSnap = np.zeros(3)   # Desired Snap (xdotdotdotdot, ydotdotdotdot, zdotdotdotdot)

        self.local_pose = PoseStamped()
        self.local_vel = TwistStamped()
        self.gravity = np.array([0, 0, 9.81])

        self.command_msg = AngularVelandAccn()
        self.max_thrust = 19.8
        self.traj = Trajectory()
        self.z_i = 0
        self.Ts = 0.01


    def pose_callback(self, msg):
        self.local_pose = msg
        self.get_logger().info('position x = {:.2f}, y = {:.2f}, z = {:.2f}'.format(self.local_pose.pose.position.x, self.local_pose.pose.position.y, self.local_pose.pose.position.z))

    def velocity_callback(self, msg):
        self.local_vel = msg


    def calculate_trajectory(self):
        current_time = self.get_clock().now().nanoseconds
        # t = (current_time - self.start_time)/1e9 # converting from nanoseconds to seconds
        # self.sDes = self.traj.desiredState(t,0.01)
        # self.desPos = np.array([self.sDes[0], self.sDes[1], self.sDes[2]])
        # self.desVel = np.array([self.sDes[3], self.sDes[4], self.sDes[5]])
        # self.desAcc = np.array([self.sDes[6], self.sDes[7], self.sDes[8]])
        # self.desThr = self.sDes[9:12]
        # self.desEul = self.sDes[12:15]
        # self.desPQR = self.sDes[15:18]
        # self.yawFF = self.sDes[18]
        # self.desJerk = self.sDes[19:22]
        # self.desSnap = self.sDes[22:25]
        self.desPos = np.array([0, 0, 3])

    def quat_mult(self,q, p):
        # Extract scalar and vector parts
        Q = np.array([[q[0], -q[1], -q[2], -q[3]],
                  [q[1],  q[0], -q[3],  q[2]],
                  [q[2],  q[3],  q[0], -q[1]],
                  [q[3], -q[2],  q[1],  q[0]]])
        return Q@p
        
    
    def quaternion_from_rotation_matrix(self, R):
    # Extract the values from R
        q0 = np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2
        q1 = (R[2, 1] - R[1, 2]) / (4 * q0)
        q2 = (R[0, 2] - R[2, 0]) / (4 * q0)
        q3 = (R[1, 0] - R[0, 1]) / (4 * q0)
        return np.array([q0, q1, q2, q3])
    
    def RotToQuat(self,R):
    
        R11 = R[0, 0]
        R12 = R[0, 1]
        R13 = R[0, 2]
        R21 = R[1, 0]
        R22 = R[1, 1]
        R23 = R[1, 2]
        R31 = R[2, 0]
        R32 = R[2, 1]
        R33 = R[2, 2]
        # From page 68 of MotionGenesis book
        tr = R11 + R22 + R33

        if tr > R11 and tr > R22 and tr > R33:
            e0 = 0.5 * np.sqrt(1 + tr)
            r = 0.25 / e0
            e1 = (R32 - R23) * r
            e2 = (R13 - R31) * r
            e3 = (R21 - R12) * r
        elif R11 > R22 and R11 > R33:
            e1 = 0.5 * np.sqrt(1 - tr + 2*R11)
            r = 0.25 / e1
            e0 = (R32 - R23) * r
            e2 = (R12 + R21) * r
            e3 = (R13 + R31) * r
        elif R22 > R33:
            e2 = 0.5 * np.sqrt(1 - tr + 2*R22)
            r = 0.25 / e2
            e0 = (R13 - R31) * r
            e1 = (R12 + R21) * r
            e3 = (R23 + R32) * r
        else:
            e3 = 0.5 * np.sqrt(1 - tr + 2*R33)
            r = 0.25 / e3
            e0 = (R21 - R12) * r
            e1 = (R13 + R31) * r
            e2 = (R23 + R32) * r

    # e0,e1,e2,e3 = qw,qx,qy,qz
        q = np.array([e0,e1,e2,e3])
        q = q*np.sign(e0)
    
        q = q/np.sqrt(np.sum(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2))
    
        return q

    def quat2Dcm(self,q):
        dcm = np.zeros([3,3])

        dcm[0,0] = q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2
        dcm[0,1] = 2.0*(q[1]*q[2] - q[0]*q[3])
        dcm[0,2] = 2.0*(q[1]*q[3] + q[0]*q[2])
        dcm[1,0] = 2.0*(q[1]*q[2] + q[0]*q[3])
        dcm[1,1] = q[0]**2 - q[1]**2 + q[2]**2 - q[3]**2
        dcm[1,2] = 2.0*(q[2]*q[3] - q[0]*q[1])
        dcm[2,0] = 2.0*(q[1]*q[3] - q[0]*q[2])
        dcm[2,1] = 2.0*(q[2]*q[3] + q[0]*q[1])
        dcm[2,2] = q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2

        return dcm

    def rotate_vector_by_quaternion(self,v, q):
        q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
        v_quat = np.array([0] + list(v))
        return self.quat_mult(self.quat_mult(q, v_quat), q_conj)[1:]

    def normalize_vector(self, v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm


    def acceleration_setpoint(self):
        vel = self.current_velocity()
        pos = self.current_position()
        desired_pos = np.array([5, 10, -10])
        desired_vel = np.array([0, 0, 0])
        pos_error = desired_pos - pos
        vel_error = desired_vel - vel
        acc_sp = np.zeros(3)
        kp_x = 0.3
        kd_x = 0.5
        kp_y = kp_x
        kd_y = kd_x
        kp_z = 1.5
        kd_z = 1.4
        ki_z = 0.2
        self.z_i += ki_z*pos_error[2]*self.Ts
        acc_sp[0] = kp_x*pos_error[0] + kd_x*vel_error[0]
        acc_sp[1] = kp_y*pos_error[1] + kd_y*vel_error[1]
        acc_sp[2] = kp_z*pos_error[2] + kd_z*vel_error[2] + self.z_i - 9.81

        return acc_sp
    
    def normalized_thrust_setpoint(self, des_accn):
        current_orientation = self.current_orientation()

        body_z = np.array([0, 0, -1])

        z_inertial = self.rotate_vector_by_quaternion(body_z,current_orientation)  
        

        des_thrust = np.dot(des_accn, z_inertial)/self.max_thrust

        if des_thrust > 1.0:
            des_thrust = 1.0
        elif des_thrust < 0.0:
            des_thrust = 0.0
        return des_thrust

    
    def desired_orientation(self, des_accn):
        # Create Full Desired Quaternion Based on Thrust Setpoint and Desired Yaw Angle
        # ---------------------------
        yaw_sp = 0.0
        # Desired body_z axis direction
        des_accn = -self.normalize_vector(des_accn)
        body_z = des_accn
        # Vector of desired Yaw direction in XY plane, rotated by pi/2 (fake body_y axis)
        
        y_C = np.array([-math.sin(yaw_sp), math.cos(yaw_sp), 0.0])
        x_C = np.array([math.cos(yaw_sp), math.sin(yaw_sp), 0.0])
        
        # Desired body_x axis direction
        body_x = np.cross(y_C, des_accn)
        body_x = self.normalize_vector(body_x)
        # Desired body_y axis direction
        body_y = np.cross(des_accn, body_x)
        body_y = self.normalize_vector(body_y)
        R_sp = np.array([body_x, body_y, body_z]).T
        qd_full = self.RotToQuat(R_sp)
        r = self.quat2Dcm(self.current_orientation())
        e_z1 = np.array([0, 0, -1])
        e_z = np.dot(r,e_z1)
        self.get_logger().info('e_z:  x: {:.2f} y: {:.2f} z: {:.2f}]'.format(e_z[0], e_z[1], e_z[2]))
        e_z_d = -self.normalize_vector(des_accn)
        qe_red = np.zeros(4)
        qe_red[0] = np.dot(e_z, e_z_d) + math.sqrt(norm(e_z)**2 * norm(e_z_d)**2)
        qe_red[1:4] = np.cross(e_z, e_z_d)
        qe_red = self.normalize_vector(qe_red)
        
        # Reduced desired quaternion (reduced because it doesn't consider the desired Yaw angle)
        self.qd_red = self.quat_mult(qe_red, self.current_orientation())

        # Mixed desired quaternion (between reduced and full) and resulting desired quaternion qd
        q_mix = self.quat_mult(self.quat_inverse(self.qd_red), qd_full)
        q_mix = q_mix*np.sign(q_mix[0])
        q_mix[0] = np.clip(q_mix[0], -1.0, 1.0)
        q_mix[3] = np.clip(q_mix[3], -1.0, 1.0)
        qd = self.quat_mult(self.qd_red, np.array([np.cos(yaw_sp*np.arccos(q_mix[0])), 0, 0, np.sin(yaw_sp*np.arcsin(q_mix[3]))]))
        
        return qd

    def quat_inverse(self,quat):
        qinv = np.array([quat[0], -quat[1], -quat[2], -quat[3]])/norm(quat)
        return qinv 

    def computeNominalReference(self):
        des_acceleration = self.acceleration_setpoint()
        des_orientation = self.desired_orientation(des_acceleration)
        roll, pitch, yaw = self.quaternion_to_euler(des_orientation)
        rollc, pitchc, yawc = self.quaternion_to_euler(self.current_orientation())

        #self.get_logger().info('current attitude:  r: {:.2f} p: {:.2f} y: {:.2f}]'.format(rollc, pitchc, yawc))
        self.get_logger().info('desired acceleration x = {:,.2f}, y = {:,.2f}, z = {:,.2f}'.format(des_acceleration[0], des_acceleration[1], des_acceleration[2])) 
        
        body_x = self.rotate_vector_by_quaternion(np.array([1, 0, 0]), des_orientation)
        body_y = self.rotate_vector_by_quaternion(np.array([0, 1, 0]), des_orientation)
        body_z = self.rotate_vector_by_quaternion(np.array([0, 0, 1]), des_orientation)
        
        reference_heading = self.desEul[2]
        q_heading = np.array([math.cos(reference_heading/2), 0, 0, math.sin(reference_heading/2)])
        x_C = self.rotate_vector_by_quaternion(np.array([1, 0, 0]), q_heading)
        y_C = self.rotate_vector_by_quaternion(np.array([0, 1, 0]), q_heading)
        
        ref_acc = self.desAcc + self.gravity
        pqr_sp = np.zeros(3)
        pqr_dot_sp = np.zeros(3)
        if(np.isclose(np.linalg.norm(ref_acc), 0)):
            pqr_sp[0] = 0.0
            pqr_sp[0] = 0.0
        else:    
            pqr_sp[0] = -1/np.linalg.norm(ref_acc)*np.dot(body_y,self.desJerk)
            pqr_sp[1] =  1/np.linalg.norm(ref_acc)*np.dot(body_x,self.desJerk)
        if(np.isclose(np.linalg.norm(np.cross(y_C,body_z)), 0)):
            pqr_sp[2] = 0.0
        else:
            pqr_sp[2] = 1/np.linalg.norm(np.cross(y_C,body_z))*(self.yawFF*np.dot(x_C,body_x) + pqr_sp[1]*np.dot(y_C,body_z))
        #print("ang rate setpoint",pqr_sp)
        thrust_dot = np.dot(body_z,self.desJerk)
        if(np.isclose(np.linalg.norm(ref_acc), 0)):
            pqr_dot_sp[0] = 0.0
            pqr_dot_sp[1] = 0.0
        else:
            pqr_dot_sp[0] = -1/np.linalg.norm(ref_acc)*(np.dot(body_y,self.desSnap) + 2*thrust_dot*pqr_sp[0] - np.linalg.norm(ref_acc)*pqr_sp[1]*pqr_sp[2])
            pqr_dot_sp[1] = -1/np.linalg.norm(ref_acc)*(np.dot(body_x,self.desSnap) + 2*thrust_dot*pqr_sp[1] - np.linalg.norm(ref_acc)*pqr_sp[0]*pqr_sp[2])
        if(np.linalg.norm(np.cross(y_C,body_z)) < 0.7):
            pqr_dot_sp[2] = 0.0
        else:    
            pqr_dot_sp[2] = 1/np.linalg.norm(np.cross(y_C,body_z))*(2*self.yawFF*pqr_sp[2]*np.dot(x_C,body_y)
                                                     - 2*self.yawFF*pqr_sp[1]*np.dot(x_C,body_z)
                                                     - pqr_sp[0]*pqr_sp[1]*np.dot(y_C,body_y)
                                                     - pqr_sp[0]*pqr_sp[2]*np.dot(y_C,body_z)
                                                     + pqr_dot_sp[1]*np.dot(y_C,body_z))
            
        self.command_msg.angular_rate.x = 0.0
        self.command_msg.angular_rate.y = 0.0
        self.command_msg.angular_rate.z = 0.0

        self.command_msg.angular_acceleration.x = pqr_dot_sp[0]
        self.command_msg.angular_acceleration.y = pqr_dot_sp[1]
        self.command_msg.angular_acceleration.z = pqr_dot_sp[2]


    def prepare_angular_vel_accn_message(self):
        # Prepare the AngularVelandAccn message
        self.calculate_trajectory()
        des_acc = self.acceleration_setpoint()
        orientation = self.desired_orientation(des_acc)
        thrust = self.normalized_thrust_setpoint(des_acc)
        self.computeNominalReference()
        self.command_msg.orientation = Quaternion(x = orientation[0], y = orientation[1], z = orientation[2], w = orientation[3])
        self.get_logger().info('desired orientation x = {:,.2f}, y = {:,.2f}, z = {:,.2f}, w = {:,.2f}'.format(orientation[0], orientation[1], orientation[2], orientation[3])) 

        self.command_msg.thrust = thrust
        # Publish the AngularVelandAccn message
        self.publisher.publish(self.command_msg)
        #self.get_logger().info('Published command Message, thrust = {:.2f}'.format(thrust))
        #self.get_logger().info('Published orientation Message, quaternion w = {:.2f}, x = {:.2f}, y = {:.2f}, z = {:.2f}'.format(orientation[0],orientation[1],orientation[2],orientation[3]))

    def quaternion_to_euler(self,quaternion):
        """
        Convert a quaternion into Euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        w = quaternion[0]
        x = quaternion[1]
        y = quaternion[2]
        z = quaternion[3]
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(t0, t1)
    
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = math.asin(t2)
    
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4)
    
        return roll*180/math.pi, pitch*180/math.pi, yaw*180/math.pi
    # from ENU to NED
    def current_orientation(self):
        q_transformation = np.array([1/math.sqrt(2), 0, 0, -1/math.sqrt(2)])
        q_received = np.array([self.local_pose.pose.orientation.w, self.local_pose.pose.orientation.x, self.local_pose.pose.orientation.y, self.local_pose.pose.orientation.z])
        q1 = self.quat_mult(q_received,q_transformation)
        q2 = np.array([q1[0], q1[2], q1[1], -q1[3]])
        return q2
    # from ENU to NED
    def current_position(self):
        q2 = np.array([self.local_pose.pose.position.y, self.local_pose.pose.position.x, -self.local_pose.pose.position.z])
        return q2
    
    def current_velocity(self):
        q2 = np.array([self.local_vel.twist.linear.y, self.local_vel.twist.linear.x, -self.local_vel.twist.linear.z])
        return q2

def main(args=None):
    rclpy.init(args=args)
    imu_publisher_subscriber = IMUPublisherSubscriber()
    executor = MultiThreadedExecutor()
    executor.add_node(imu_publisher_subscriber)
    executor.spin()
    imu_publisher_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
