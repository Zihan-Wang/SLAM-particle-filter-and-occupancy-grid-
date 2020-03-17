import numpy as np
import load_data as ld
import math
import matplotlib.pyplot as plt
import mapping
import seaborn as sns

# dead reckoning with one particle and noise free
NP = 1

# motion model without noise
def motion_model(x, u):
    return x + u

# predict
def dr_prediction(px,u):
    return motion_model(px, u)

if __name__ == "__main__":
    # px is the particle
    px = np.zeros((3,1))
    l0 = ld.get_lidar('lidar/train_lidar2')
    j0 = ld.get_joint("joint/train_joint2")
    l0_len = len(l0)

    # px_h record Motion trajectory
    px_h = px

    # init control input u
    u = px

    # init map
    map = np.zeros((1201, 1201))

    # U is the accumulated control input
    U = mapping.get_body_pose_inWorld(l0)

    # every 10 data, I choose one
    segment = 40

    for i in range(int(l0_len/segment)):
        # get control input
        U[:, i * segment:i * segment + 1]
        delta_u = U[:, i * segment:i * segment + 1] - u
        u = U[:, i * segment:i * segment + 1]

        # predict
        px = dr_prediction(px, delta_u)

        # update log_odds
        map += mapping.get_Occmap(l0, j0, -30, 30, -30, 30, 0.05, i*segment, px)

        #update motion trajectory
        px_h = np.hstack((px_h, px))

    np.save("dr_pose2.npy",px_h)
    np.save("drmap2.npy",map)
    plt.plot(np.array(px_h[0, :]).flatten(),
                     np.array(px_h[1, :]).flatten(), "-b")
    plt.axis("equal")
    plt.grid(True)
    plt.show()
    sns.heatmap(map, center=0, vmin=-20, vmax=20)
    plt.show()
