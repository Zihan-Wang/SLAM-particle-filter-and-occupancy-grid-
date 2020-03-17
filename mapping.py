import numpy as np
import load_data as ld
import math
import p2_utils as p2
import matplotlib.pyplot as plt
import seaborn as sns
Z_pose = 1.41

# simply accumulate "delta_pose" and return accumulated pose
def get_body_pose_inWorld(l0):
    l0_length = len(l0)
    body_Wpose = np.zeros((3, l0_length))
    for i in range(l0_length):
        body_Wpose[0, i] = body_Wpose[0, i - 1] + l0[i]['delta_pose'][0][0]
        body_Wpose[1, i] = body_Wpose[1, i - 1] + l0[i]['delta_pose'][0][1]
        body_Wpose[2, i] = body_Wpose[2, i - 1] + l0[i]['delta_pose'][0][2]
    return body_Wpose

# Z Transform Matrix for body frame to world frame
def get_body2world_TMatrix(yaw):
    Z_T = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                    [math.sin(yaw), math.cos(yaw), 0],
                    [0, 0, 1]])
    return Z_T

# head frame To body frame
def get_lindar2bodyframe_TMatrix(neck_angle, head_angle):

    # Y transform Matrix
    Y_T = np.array([[math.cos(head_angle), 0, math.sin(head_angle)],
                    [0, 1, 0],
                    [-math.sin(head_angle), 0, math.cos(head_angle)]])

    # Z transform Matrix
    Z_T = np.array([[math.cos(neck_angle), -math.sin(neck_angle), 0],
                    [math.sin(neck_angle), math.cos(neck_angle), 0],
                    [0, 0, 1]])
    return Z_T.dot(Y_T)

# use above function, we can get laser pose in world:
# l0, j0 are data sets
# laser_index is the index in data set l0
# body_Wpose is the body pose in world frame, it can be used to calculate laserpoint's pose in world frame
def get_laser_pose_inWorld(l0, j0, laser_index, body_Wpose):
    ranges = l0[laser_index]['scan'].T
    angles = np.array([np.arange(-135, 135.25, 0.25) * np.pi / 180.]).T

    # get rid of points which are too far, too close or hit ground
    indValid = np.logical_and((ranges < 30), (ranges > 0.1))
    ranges = ranges[indValid]
    angles = angles[indValid]

    # xy position in the sensor frame
    xs0 = ranges * np.cos(angles)
    ys0 = ranges * np.sin(angles)
    zs0 = np.zeros((len(xs0), 1))

    # find the same absolute timestamp for data in l0 and j0
    scan_time = np.abs(j0['ts'][0] - l0[laser_index]['t'][0][0]).argmin()

    # to get laserpoints in world frame
    laser_Wpose = get_lindar2bodyframe_TMatrix(j0['head_angles'][0][scan_time], j0['head_angles'][1][scan_time]).dot(np.c_[xs0, ys0, zs0].T)
    laser_Wpose[2] = laser_Wpose[2] + Z_pose
    laser_Wpose = get_body2world_TMatrix(body_Wpose[2]).dot(laser_Wpose)
    laser_Wpose[0,:] += body_Wpose[0]
    laser_Wpose[1,:] += body_Wpose[1]

    # get rid of points which are too far, too close or hit ground
    laserValid = np.logical_and((laser_Wpose[2] < 5), (laser_Wpose[2] > 0.1))
    laser_Wpose = laser_Wpose[:, laserValid]
    return laser_Wpose

# get occupancy grid map
def get_Occmap(l0, j0, xmin, xmax, ymin, ymax, res, laser_index, body_Wpose):

    # get laser points in world frame
    laser_Wpose = get_laser_pose_inWorld(l0, j0, laser_index, body_Wpose)

    # measurements of valid laser points
    map_meas = len(laser_Wpose[0])

    # init map
    map = np.zeros((int((ymax - ymin)/res+1),int((xmax - xmin)/res+1)))

    for i in range(map_meas):
        # transform world frame to map frame
        lx = (np.ceil((laser_Wpose[0][i] - xmin) / 0.05)).astype(np.int16)-1
        ly = (np.ceil((laser_Wpose[1][i] - ymin) / 0.05)).astype(np.int16)-1
        bx = (np.ceil((body_Wpose[0][0] - xmin) / 0.05)).astype(np.int16)-1
        by = (np.ceil((body_Wpose[1][0] - ymin) / 0.05)).astype(np.int16)-1

        # use bresenham algorithm to get free cells
        r = p2.bresenham2D(bx, by, lx, ly)

        # log-odds
        map[lx, ly] += np.log(4)
        map[r[0].astype(np.int16), r[1].astype(np.int16)] += np.log(0.8)

    return map

if __name__ == "__main__":
    l0 = ld.get_lidar("lidar/train_lidar2")
    j0 = ld.get_joint("joint/train_joint2")
    body_Wpose = get_body_pose_inWorld(l0)
    map = get_Occmap(l0, j0, -20, 20, -20, 20, 0.05, 0, body_Wpose[:,0:1])
    map_index = np.where(map > 0)
    map_indexn = np.where(map < 0)
    map_vi = np.zeros(map.shape)
    map_vi[map_index] = 1
    map_vi[map_indexn] = 0
    plt.imshow(map_vi, cmap="hot")
    plt.title("Occupancy map")
    plt.xlabel('y--0.05m/cell')
    plt.ylabel('x--0.05m/cell')
    plt.show()
    sns.heatmap(map, center=0, vmin=-5, vmax=5)
    plt.xlabel('y--0.05m/cell')
    plt.ylabel('x--0.05m/cell')
    plt.show()
