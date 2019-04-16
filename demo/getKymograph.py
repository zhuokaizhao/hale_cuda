# The program generates kymograph based on tracked pioneer neuron
# Author: Zhuokai Zhao
# Contact: zhuokai@uchicago.edu

from __future__ import print_function
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def print_help():
    print('''    
    This program generates kymograph based on tracked pioneer neuron locations
    
    Usage:
    python/pythonw getKymograph.py [tracked_file -- default coord_newtrack_pioneer_scale_old.txt]''')

# helper fucntion that computes curvatures
def computeCurvature(x, y, z):
    # construct Fernet frame and calculate curvature k
    # get derivatives alone x, y and z with respect to time
    x_d = np.gradient(x)
    y_d = np.gradient(y)
    z_d = np.gradient(z)
    # r_d is the velocity, consistent with Fernet frame notation
    r_d = np.zeros((x_d.size, 3))
    for i in range(x_d.size):
        r_d[i, 0] = x_d[i]
        r_d[i, 1] = y_d[i]
        r_d[i, 2] = z_d[i]

    # find the unit tangent vector T, which is pointing in the same direction as v and is a unit vector
    r_d_norm = np.sqrt(x_d*x_d + y_d*y_d + z_d*z_d)
    T = np.array([1/abs(r_d_norm)]).transpose() * r_d
    Tx = T[:, 0]
    Ty = T[:, 1]
    Tz = T[:, 2]

    # find the norm of the derivative of T, which is ||T'(t)||
    Tx_d = np.gradient(Tx)
    Ty_d = np.gradient(Ty)
    Tz_d = np.gradient(Tz)
    T_d = np.zeros((Tx_d.size, 3))
    for i in range(Tx_d.size):
        T_d[i, 0] = Tx_d[i]
        T_d[i, 1] = Ty_d[i]
        T_d[i, 2] = Tz_d[i]

    T_d_norm = np.sqrt(Tx_d * Tx_d + Ty_d * Ty_d + Tz_d * Tz_d)

    # curvature k = ||T'(t)|| / ||r'(t)||
    curvature = T_d_norm / r_d_norm

    return curvature

# Helper function that adds Gaussian noise to pioneer locations
def addGaussianNoise(loc, sigma):
    # get the information about the image
    mean = 0
    gauss = np.random.normal(mean, sigma, loc.size)
    loc_noisy = loc + gauss
    
    return loc_noisy

# Helper function that computes the root mean squared error between two sets of curvatures
def computeRMS(cur1, cur2):
    rms = np.sqrt(np.sum((cur1 - cur2) ** 2))
    # we assume that two images have the same dimensions
    rms /= float(cur1.size)

    return rms

# helper function that performs 3D scatter plot with annotation
def plot3D(i, time, x, y, z, myTitle):
    fig = plt.figure(i)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    # make the time stamp label
    for i, stamp in enumerate(time):
        label = str(stamp)
        ax.text(x[i], y[i], z[i], label)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.title(myTitle)

def main(argv):
    print_help()
    filename = argv[0] if len(argv) > 0 else 'coord_newtrack_pioneer_scale_old.txt'
    timeShift = float(argv[1]) if len(argv) > 1 else 5
    sigma = float(argv[2]) if len(argv) > 2 else 1
    
    # load the txt file
    with open(filename) as textFile:
        allPositions_pioneer = [line.split() for line in textFile]

    # we right now only have pioneer data
    allPositions_pioneer = np.asarray(allPositions_pioneer)
    # change from string to float
    allPositions_pioneer = allPositions_pioneer.astype(float)
    time_pioneer = allPositions_pioneer[:,0]
    x_pioneer = allPositions_pioneer[:,1]
    y_pioneer = allPositions_pioneer[:,2]
    z_pioneer = allPositions_pioneer[:,3]

    # 3D scatter plot of pioneers
    plot3D(1, time_pioneer, x_pioneer, y_pioneer, z_pioneer, 'Pioneer locations vs t')

    # make synthetic followers data
    allPositions_follower = allPositions_pioneer
    # all time stamp plus 10
    time_follower = allPositions_follower[:,0] + timeShift
    x_follower = allPositions_follower[:,1] + addGaussianNoise(x_pioneer, sigma)
    y_follower = allPositions_follower[:,2] + addGaussianNoise(y_pioneer, sigma)
    z_follower = allPositions_follower[:,3] + addGaussianNoise(z_pioneer, sigma)

    # 3D scatter plot of followers
    plot3D(2, time_follower, x_follower, y_follower, z_follower, 'Follower locations vs t')
    

    # get the curvature of both pioneer and follower
    curvature_pioneer = computeCurvature(x_pioneer, y_pioneer, z_pioneer)
    curvature_follower = computeCurvature(x_follower, y_follower, z_follower)

    # generate kymograph
    plt.figure(3)
    plt.plot(time_pioneer, curvature_pioneer)
    plt.plot(time_follower, curvature_follower)
    plt.legend(('Pioneer', 'Follower'))
    myTitle = 'Pioneer(real) and follower(synthetic) neurons positions vs time\nRMS =' + str(computeRMS(curvature_pioneer, curvature_follower))
    plt.title(myTitle)
    plt.xlabel('t')
    plt.ylabel('Curvature')

    # generate the RMS plot as noise level goes up
    allRMS = []
    allSigma = []
    for i in range(10000):
        sigma = sigma + 0.01
        allSigma.append(sigma)
        curSumRMS = 0
        for j in range(10):
            # make synthetic followers data
            allPositions_follower = allPositions_pioneer
            # all time stamp plus 10
            time_follower = allPositions_follower[:,0] + timeShift
            x_follower = allPositions_follower[:,1] + addGaussianNoise(x_pioneer, sigma)
            y_follower = allPositions_follower[:,2] + addGaussianNoise(y_pioneer, sigma)
            z_follower = allPositions_follower[:,3] + addGaussianNoise(z_pioneer, sigma)

            curvature_follower = computeCurvature(x_follower, y_follower, z_follower)
            curSumRMS = curSumRMS + computeRMS(curvature_pioneer, curvature_follower)
        
        # get the average of sum
        curRMS = float(curSumRMS/10.)
        allRMS.append(curRMS)

    # plot the RMS vs sigma
    plt.figure(4)
    plt.plot(allSigma, allRMS)
    myTitle = 'Root squared mean error as sigma increases'
    plt.title(myTitle)
    plt.xlabel('Sigma')
    plt.ylabel('RMS')
    
    
    plt.show()




if __name__ == "__main__":
    main(sys.argv[1:])
