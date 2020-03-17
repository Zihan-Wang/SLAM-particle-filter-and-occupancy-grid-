import numpy as np
import load_data as ld
import p2_utils as p2
import mapping as mapping
import prediction as prediction
import scipy.special as ss
import math
import matplotlib.pyplot as plt
import seaborn as sns

# get corr-matrix and adjust particles according to the max corr
def get_corr(l0, j0, particle, i, map):

    # get 0-1 map
    map_inv = np.zeros(map.shape)
    map_inv[np.where(map > 0)] = 1
    map_inv[np.where(map < 0)] = 0

    #if i % 2000 == 0:
    #    plt.imshow(map_inv)
    #    plt.show()

    # number of particle
    shape = np.shape(particle)
    Np = shape[1]

    # init corr-matrix
    C = np.zeros((1, Np))

    for j in range(Np):
        laser_pose_inWorld = mapping.get_laser_pose_inWorld(l0, j0, i, particle[:,j:j+1])
        x_im = np.arange(-30, 30 + 0.05, 0.05)  # x-positions of each pixel of the map
        y_im = np.arange(-30, 30 + 0.05, 0.05)  # y-positions of each pixel of the map
        x_range = np.arange(-0.2, 0.2 + 0.05, 0.05) #physical x,y,positions you want to evaluate "correlation"
        y_range = np.arange(-0.2, 0.2 + 0.05, 0.05) #physical x,y,positions you want to evaluate "correlation"


        c = p2.mapCorrelation(map_inv, x_im, y_im, laser_pose_inWorld[0:3, :], x_range, y_range) # get corr of physical x,y,positions you want to evaluate "correlation"
        corr_new = np.max(c) # value of max corr
        x_index = np.where(c == np.max(c)) # index of the max corr

        # adjust particle according the index of max corr
        x_index = np.sum(x_index, axis=1)/len(x_index[0])
        x_index = np.array([[x_index[0]], [x_index[1]]])
        x_new = (x_index - 4)*0.05
        particle[0:2, j:j+1] += x_new

        C[0][j] = corr_new

    # get observation function pz
    pz = ss.softmax(C)
    return particle, pz

def update(l0, j0, particle, weight, i, map):
    particle, pz = get_corr(l0, j0, particle, i, map)

    # update by observation pz
    weight = np.multiply(weight, pz)/weight.dot(pz.T)

    # get position by particles and weight
    px = particle.dot(weight.T)
    return px, particle, weight


# low variance re-sampling
def resampling(particle, weight):

    Np = np.shape(particle)[1]

    # to get effective number of particle
    Neff = 1.0 / (weight.dot(weight.T))

    # I set threshold as 14
    if Neff < 14:

        # cumulated weights, ready to compare
        cumlatedWeight = np.array([np.cumsum(weight)])
        resampleid = np.array([np.cumsum(weight * 0.0 + 1 / Np) - 1 / Np]) + np.random.rand(Np) / Np

        # init indexes of replaced particles
        inds = []
        ind = 0

        # find index of weight almost zeros, then replaced by other particle
        for ip in range(Np):

            # if weight small, find next index
            while resampleid[0, ip] > cumlatedWeight[0, ind]:
                ind += 1

            # use this index
            inds.append(ind)

        particle = particle[:, inds]

        # init weight for resampled particles
        weight = np.zeros((1, Np)) + 1.0 / Np

    return particle, weight


if __name__ == "__main__":
    l0 = ld.get_lidar('lidar/train_lidar4')
    l0_len = len(l0)
    U = mapping.get_body_pose_inWorld(l0)

    # use one data every 20 data
    segment = 15
    j0 = ld.get_joint('joint/train_joint4')

    # 25 particles
    Np = 25

    #init particles, weights, map, control input, pose and motion trjectory
    particle = np.zeros((3, Np))
    weight = np.ones((1, Np)) / Np
    px = U[:, 0:1]
    px_h = px
    particle_pose_timestamp = l0[0]['t'][0][0]
    map = np.zeros((1201, 1201))
    u = np.zeros((3, 1))

    for i in range(int(l0_len/segment)):

        # get control input
        delta_u = U[:, i*segment:i*segment+1] - u
        u = U[:, i*segment:i*segment+1]

        # predict particles
        particle = prediction.prediction(particle, delta_u)

        # find occupancy grid map and update log-odds
        map += mapping.get_Occmap(l0, j0, -30, 30, -30, 30, 0.05, i * segment, px)

        # update pose and particles by log-odds
        px, particle, weight = update(l0, j0, particle, weight, i*segment, map)

        # find if we need to resample
        particle, weight = resampling(particle, weight)

        # record trajectory
        particle_pose_timestamp = np.hstack((particle_pose_timestamp, l0[i*segment]['t'][0][0]))
        px_h = np.hstack((px_h, px))
    #np.save("particle_pose4.npy", px_h)
    #np.save("particle_time4.npy", particle_pose_timestamp)
    #np.save("occu_map4.npy", map)
    sns.heatmap(map, center=0, vmin=-20, vmax=20)
    plt.plot(np.array(((px_h[1, :] / 0.05 + 600).astype(np.int16))).flatten(),
             np.array(px_h[0, :] / 0.05 + 600).astype(np.int16).flatten(), "-k")
    plt.show()


