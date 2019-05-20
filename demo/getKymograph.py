# The program generates kymograph based on tracked pioneer neuron
# Author: Zhuokai Zhao
# Contact: zhuokai@uchicago.edu

from __future__ import print_function
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from scipy import fftpack, ndimage, signal
import math
from sklearn import preprocessing

def print_help():
    print('''    
    This program generates kymograph based on tracked pioneer neuron locations
    
    Usage:
    python/pythonw getKymograph.py [mode -- synthetic or real, default synthetic] 
                                   [tracked_file -- default coord_newtrack_pioneer_scale_old.txt]
                                   [time shift -- default 0]
                                   [noise std sigma_noise -- default 1]
                                   [smooth std sigma_smooth -- default 10]''')


# Helper function that computes derivative
# A is in three-dimension, first column is x, second column is y, third column is z
def computeDerivative(A):
    Ax = A[:, 0]
    Ay = A[:, 1]
    Az = A[:, 2]
    Ax_d = np.gradient(Ax)
    Ay_d = np.gradient(Ay)
    Az_d = np.gradient(Az)
    A_d = np.zeros((Ax_d.size, 3))
    A_d[:, 0] = Ax_d
    A_d[:, 1] = Ay_d
    A_d[:, 2] = Az_d

    return A_d

# helper function that computes pseudo side vector of sample points
def computePseudoSideVector(allPoints):
    pca = PCA(n_components=3)
    pca.fit(allPoints)
    # choose the second as pseudo side vector
    S = pca.components_[1]

    return S

# Helper function that computes norm
def computeNorm(A):
    sum = np.zeros(A.shape[0])
    for i in range(A.shape[1]):
        sum = sum + A[:, i]*A[:, i]

    norm = np.array([np.sqrt(sum)]).transpose()

    return norm

# helper fucntion that computes curvatures
def computeCurvature(allPoints):
    # when the case data has time stamps
    if (allPoints.shape[1] == 4):
        newAllPoints = np.zeros[allPoints.shape[0], 3]
        newAllPoints[:, :] = allPoints[:, 1:]
        r_d = computeDerivative(newAllPoints)

    # r_d is the velocity
    r_d = computeDerivative(allPoints)
    # norm of r_d
    r_d_norm = computeNorm(r_d)

    # T is nx3 matrix, each row corresponds to a point
    T = 1/abs(r_d_norm) * r_d
    # Compute the derivative of T
    T_d = computeDerivative(T)
    # norm of T, ||T||
    T_d_norm = computeNorm(T_d)

    # curvature k = ||T'(t)|| / ||r'(t)||
    curvature = T_d_norm / r_d_norm

    return curvature

# helper function that computes the new characteristic we come up with
def computeNewCharacter1(allPoints):
    # when the case data has time stamps
    if (allPoints.shape[1] == 4):
        newAllPoints = np.zeros[allPoints.shape[0], 3]
        newAllPoints[:, :] = allPoints[:, 1:]
        r_d = computeDerivative(newAllPoints)
    else:
        # r_d is the velocity
        r_d = computeDerivative(allPoints)

    # find the unit tangent vector T, which is pointing in the same direction as v and is a unit vector
    # r_d_norm is nx3
    r_d_norm = computeNorm(r_d)

    # T is nx3 matrix, each row corresponds to a point
    T = 1/abs(r_d_norm) * r_d
    
    # compute pseudo side vector S, which is also nx3
    S = computePseudoSideVector(allPoints)

    # construct U matrix (unnormalized), U = S cross T
    U = np.cross(S, T)
    # normalize U
    U_norm = computeNorm(U)
    U = 1/abs(U_norm) * U

    # gradient of U
    U_d = computeDerivative(U)
    # compute norm of U_d
    U_d_norm = computeNorm(U_d)

    # new characteristic
    newK = U_d_norm / r_d_norm

    return newK

# this does not really make sense
# def computeNewCharacter2(allPoints):
#     # when the case data has time stamps
#     if (allPoints.shape[1] == 4):
#         newAllPoints = np.zeros[allPoints.shape[0], 3]
#         newAllPoints[:, :] = allPoints[:, 1:]
#         r_d = computeDerivative(newAllPoints)

#     # r_d is the velocity
#     r_d = computeDerivative(allPoints)
#     # find the unit tangent vector T, which is pointing in the same direction as v and is a unit vector
#     # r_d_norm is nx3
#     r_d_norm = computeNorm(r_d)

#     # T is nx3 matrix, each row corresponds to a point
#     T = 1/abs(r_d_norm) * r_d
    
#     # compute pseudo side vector S, which is also nx3
#     S = computePseudoSideVector(allPoints)

#     # construct U matrix (unnormalized), U = S cross T
#     U = np.cross(S, T)

#     # normalize U
#     U_norm = computeNorm(U)
#     U = 1/abs(U_norm) * U

#     # gradient of U
#     U_d = computeDerivative(U)
#     # compute norm of U_d
#     U_d_norm = computeNorm(U_d)

#     # compute L = T cross U
#     L = np.cross(T, U)
#     L_norm = computeNorm(L)
#     # normalized L
#     L = 1/abs(L_norm) * L

#     # compute the derivative of L
#     L_d = computeDerivative(L)
#     # norm of L_d
#     L_d_norm = computeNorm(L_d)

#     # new K = sqrt(U_d_norm^2 + L_d_norm^2) / r_d_norm^2
#     newK = np.sqrt(U_d_norm*U_d_norm + L_d_norm*L_d_norm) / r_d_norm

#     return newK


# Helper function that adds Gaussian noise to pioneer locations
def addGaussianNoise(loc, mean, sigma):
    # get the information about the image
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

# helper function that computes the Euclidean distance between two points
def distance(p1, p2):
    dist = 0
    # dimension check
    if (len(p1) != len(p2)):
        print('Could not compute distance between p1 and p2, incompatible dimension')
        print('p1 has length ' + str(len(p1)))
        print('p2 has length ' + str(len(p2)))
        return dist

    sum = 0
    for i in range(len(p1)):
        sum = sum + (p1[i]-p2[i])**2
    
    dist = np.sqrt(sum)

    return dist

# helper function that is the Gaussian kernel
def gaussian(x, sigma):
    g = 1.0/(2 * math.pi * (sigma**2)) * math.exp(-(x**2)/(2*sigma**2))
    return g

# helper function that performs the customized Bilateral filter
# it uses the new characteristic newK1 as the "pixel intensity"
# filter the newK1 and recover points from newK1
# def customized_Bilateral(allPoints_noised, allFeatures, diameter, sigma_dist, sigma_feature):
#     # initialize the filtered result
#     allPoints_filtered = np.zeros(allPoints_noised.shape)
#     allFeatures_filterd = np.zeros(allFeatures.shape)
#     curFiltered = 0

#     # iterate through all the points
#     for t in range(len(allPoints_noised)):
#         curPoint = allPoints_noised[t]
#         wp = 0

#         # radius, which is used in the loop 
#         radius = diameter/2
#         for i in range(diameter):
#             for j in range(diameter):
#                 # just go from -radius to +radius
#                 neighbor_t = t - (radius - i)
#                 if neighbor_t >= len(allPoints_noised):
#                     # don't apply in this case
#                     continue
                
#                 # get the curNeighbor point
#                 curNeighbor = allPoints_noised[neighbor_t]
#                 # two gaussians
#                 g_dist = gaussian(distance(curPoint, curNeighbor), sigma_dist)
#                 g_feat = gaussian(allFeatures[neighbor_t] - allFeatures[t], sigma_feature)
#                 # Bilateral filter is the multiplication of the two
#                 bilateral = g_dist * g_feat
#                 curFiltered = curFiltered + allFeatures[neighbor_t] * bilateral
#                 wp = wp + bilateral
        
#         # normalize
#         curFiltered = curFiltered / wp
#         allFeatures_filterd[t] = curFiltered

# new method that treats the third dimension as the intensity in bilateral filter
# index is the column index of the allPoints_noised that we want to treat as intensity
def customized_Bilateral(allPoints_noised, index, diameter, sigma_dist, sigma_feature):
    # get the input ready
    # if we are picking the first column as intensity
    if (index == 0):
        allPoints = allPoints_noised[:, 1:]
    elif (index == 1):
        # all the columns to the left of choses index column
        allPoints1 = allPoints_noised[:, 0:index]
        # all the columns to the right of choses index column
        allPoints2 = allPoints_noised[:, index+1:]
        # concatenate together
        allPoints = np.zeros((len(allPoints_noised), 2))
        for i in range(len(allPoints)):
            allPoints[i, 0] = allPoints1[i]
            allPoints[i, 1] = allPoints2[i]
    elif (index == 2):
        allPoints = allPoints_noised[:, 0:index]

    # now we treat the index-column as the intensity
    intensity = allPoints_noised[:, index]

    # initialize the filtered result
    intensityFiltered = np.zeros(intensity.shape)
    curFiltered = 0

    # iterate through all the points
    for t in range(len(allPoints)):
        curPoint = allPoints[t, :]
        wp = 0

        # radius, which is used in the loop 
        radius = int(diameter/2)
        for i in range(diameter):
            # just go from -radius to +radius
            neighbor_t = t - (radius - i)
            # boundary check
            if neighbor_t >= len(allPoints_noised):
                continue
            if neighbor_t <= 0:
                continue
            
            # get the curNeighbor point
            curNeighbor = allPoints[neighbor_t, :]
            # two gaussians
            # use the distance between as input to a Gaussian, assumed zero mean
            g_dist = gaussian(distance(curPoint, curNeighbor), sigma_dist)
            # use the intensity difference as input to a Gaussian, assumed zero mean
            g_feat = gaussian((intensity[neighbor_t] - intensity[t]), sigma_feature)
            # Bilateral filter is the multiplication of the two
            bilateral = g_dist * g_feat
            # sum up the filtered value of the current near neighbor
            curFiltered = curFiltered + intensity[neighbor_t] * bilateral
            # sum up the filter value of the current neighbor
            wp = wp + bilateral
        
        # weighted average of intensity values from nearby pixels
        curFiltered = curFiltered / wp
        intensityFiltered[t] = curFiltered

        # now put it back
        # initialize output
        allPoints_filtered = np.zeros(allPoints_noised.shape)
        for i in range(len(allPoints_filtered)):
            if (index == 0):
                allPoints_filtered[i, 0] = intensity[i]
                allPoints_filtered[i, 1] = allPoints[i, 0]
                allPoints_filtered[i, 2] = allPoints[i, 1]
            elif (index == 1):
                # all the columns to the left of choses index column
                allPoints1 = allPoints[:, 0]
                # all the columns to the right of choses index column
                allPoints2 = allPoints[:, 1]
                # concatenate together
                allPoints_filtered[i, 0] = allPoints1[i]
                allPoints_filtered[i, 1] = intensityFiltered[i]
                allPoints_filtered[i, 2] = allPoints2[i]
            elif (index == 2):
                allPoints_filtered[i, 0] = allPoints[i, 0]
                allPoints_filtered[i, 1] = allPoints[i, 1]
                allPoints_filtered[i, 2] = intensity[i]

        # return the smoothed points
        return allPoints_filtered

# 3D bilateral filter
# Input:
#   current point p_cur
#   previous point p_prev
#   sigma for distance
#   sigma for intensity
def bilateral3D(p_cur, p_prev, d, sigma_dist, sigma_intensity):
    q_new = 0

    return q_new


# the function implements the algorithm explained in the PDF
def novelDenoise(allPoints_noised, sigma_dist, sigma_intensity, d):
    # first we find the PCA of the noised points
    pca = PCA(n_components=3)
    pca.fit(allPoints_noised)
    # choose the second as pseudo side vector
    v1_vect = pca.components_[0]
    v2_vect = pca.components_[1]
    v3_vect = pca.components_[2]

    # then iterate all points in the path, starting with the 2nd
    for i in range(1, len(allPoints_noised)):
        # current point and previous point
        p_cur = allPoints_noised[i, :]
        p_prev = allPoints_noised[i-1, :]
        # construct vector between the two points
        p_vect = p_cur - p_prev
        # move the PCA vector to be rooted the same as p_vect
        v1_cur_vect = p_prev + v1_vect
        v2_cur_vect = p_prev + v2_vect
        v3_cur_vect = p_prev + v3_vect
        # project p_vect onto these PCA vectors, which are vectors with respect to p_prev
        p_v1_vect = np.dot(p_vect, v1_cur_vect) * preprocessing.normalize(v1_cur_vect)
        p_v2_vect = np.dot(p_vect, v2_cur_vect) * preprocessing.normalize(v2_cur_vect)
        p_v3_vect = np.dot(p_vect, v3_cur_vect) * preprocessing.normalize(v3_cur_vect)
        # construt q vector, which is perpendicular to v_cur_vect
        q_v1_vect = p_vect - p_v1_vect
        q_v2_vect = p_vect - p_v2_vect
        q_v3_vect = p_vect - p_v3_vect
        # since all p_v#_vect are all rooted at p_prev, we transform it to be based on Euclidean origin
        p_v1 = p_prev + p_v1_vect
        p_v2 = p_prev + p_v2_vect
        p_v3 = p_prev + p_v3_vect
        # norm of q vectors
        q_v1 = np.linalg.norm(q_v1_vect)
        q_v2 = np.linalg.norm(q_v2_vect)
        q_v3 = np.linalg.norm(q_v3_vect)
        
        # now we've had everything, we do denoising
        q_v1_new = bilateral3D(p_cur, p_prev, d, sigma_dist, sigma_intensity)


# helper function that draws a helix
def drawHelix(theta_max, numPoints):
    theta = np.linspace(0, theta_max, numPoints)
    x =  np.cos(theta)
    y =  np.sin(theta)
    z = theta
    allPoints = np.zeros((len(theta), 3))
    allPoints[:, 0] = x
    allPoints[:, 1] = y
    allPoints[:, 2] = z

    return allPoints, theta

def main(argv):
    print_help()
    mode = argv[0] if len(argv) > 0 else 'synthetic'
    filename = argv[1] if len(argv) > 1 else 'coord_newtrack_pioneer_scale_old.txt'
    timeShift = float(argv[2]) if len(argv) > 2 else 0
    sigma_noise = float(argv[3]) if len(argv) > 3 else 0.1
    sigma_smooth = float(argv[4]) if len(argv) > 4 else 10

    # if we want all individual plots
    plotIndividual = False

    # synthetic helix dataset
    if mode == 'synthetic':
        # number of data points that we are generating
        n = 100

        # Plot a right-hand helix along the z-axis
        theta_max = 8 * np.pi
        allPoints, theta = drawHelix(theta_max, n)

        # add noise to current points
        allPoints_noised = np.zeros((len(allPoints), 3))
        allPoints_noised[:, 0] = addGaussianNoise(allPoints[:,0], 0, sigma_noise)
        allPoints_noised[:, 1] = addGaussianNoise(allPoints[:,1], 0, sigma_noise)
        allPoints_noised[:, 2] = addGaussianNoise(allPoints[:,2], 0, sigma_noise)

        # filter the noise with Gaussian filter
        allPoints_noised_gauss = np.zeros((len(allPoints), 3))
        allPoints_noised_gauss[:, 0] = ndimage.gaussian_filter1d(allPoints_noised[:, 0], sigma_smooth)
        allPoints_noised_gauss[:, 1] = ndimage.gaussian_filter1d(allPoints_noised[:, 1], sigma_smooth)
        allPoints_noised_gauss[:, 2] = ndimage.gaussian_filter1d(allPoints_noised[:, 2], sigma_smooth)

        # filter the noise with this new method
        allPoints_noised_novel = customized_Bilateral(allPoints_noised, 1, 10, 1, 1)

        # curvature of original dataset
        curvature = computeCurvature(allPoints)
        # get the curvature of synthetic dataset after Gaussian smooth
        curvature_noised_gauss = computeCurvature(allPoints_noised_gauss)
        # also gaussian smooth the curvature result
        curvature_noised_gauss = ndimage.gaussian_filter1d(curvature_noised_gauss, sigma_smooth)
        # get the curvature of synthetic dataset after my smooth
        curvature_noised_my = computeCurvature(allPoints_noised_novel)
        # also gaussian smooth the my result
        curvature_noised_my = ndimage.gaussian_filter1d(curvature_noised_my, sigma_smooth)
        
        # newK1 of the original dataset
        newK1 = computeNewCharacter1(allPoints)
        # get the newK1 of the synthetic dataset after Gaussian smooth
        newK1_noised_gauss = computeNewCharacter1(allPoints_noised_gauss)
        # also gaussian smooth the newK1 result
        newK1_noised_gauss = ndimage.gaussian_filter1d(newK1_noised_gauss, sigma_smooth)
        # get the newK1 of the synthetic dataset after my smooth
        newK1_noised_my = computeNewCharacter1(allPoints_noised_novel)
        # also gaussian smooth the newK1 result
        newK1_noised_my = ndimage.gaussian_filter1d(newK1_noised_my, sigma_smooth)
        

        # # newK2 of original dataset
        # newK2 = computeNewCharacter2(allPoints)
        # # get the newK2 of the synthetic dataset after Gaussian smooth
        # newK2_noised_gauss = computeNewCharacter2(allPoints_noised_gauss)
        # # also gaussian smooth the newK2 result
        # newK2_noised_gauss = ndimage.gaussian_filter1d(newK2_noised_gauss, sigma_smooth)
        # # get the newK2 of the synthetic dataset after Gaussian smooth
        # newK2_noised_my = computeNewCharacter2(allPoints_noised_my)
        # # also gaussian smooth the newK2 result
        # newK2_noised_my = ndimage.gaussian_filter1d(newK2_noised_my, sigma_smooth)



        '''
        fig1 to fig4 are individual plots, fig 5 is all combined plots
        '''
        if (plotIndividual):
            # make the plot
            fig1 = plt.figure(1, figsize=(8, 6))
            ax1 = fig1.add_subplot(111, projection='3d')
            ax1.plot(allPoints[:,0], allPoints[:,1], allPoints[:,2], 'b', lw=2)
            ax1.plot(allPoints_noised_gauss[:, 0], allPoints_noised_gauss[:, 1], allPoints_noised_gauss[:, 2], 'r', lw=2)
            ax1.plot(allPoints_noised_novel[:, 0], allPoints_noised_novel[:, 1], allPoints_noised_novel[:, 2], 'r', lw=2)
            # plot the pseudo-side vector
            # get the pseudo side vector first
            S = computePseudoSideVector(allPoints)
            origin = [0, 0, 0]
            X, Y, Z = zip(origin)
            U, V, W = zip(S)
            ax1.quiver(X, Y, Z, U, V, W, color='k')
            
            # generate three kinds of kymograph
            # curvature based
            fig2 = plt.figure(2, figsize=(8, 6))
            ax2 = fig2.add_subplot(111)
            ax2.plot(theta[2:len(theta)-2], curvature[2:len(curvature)-2])
            plt.plot(theta[2:len(theta)-2], curvature_noised_gauss[2:len(curvature_noised_gauss)-2])
            plt.legend(('Original', 'Noised'))
            myTitle = 'Synthetic helix curvature (||T_d(t)|| / ||r_d(t)||) vs time\nSigma=' + str(sigma_noise) + '\nRMS =' + str(computeRMS(curvature, curvature_noised_gauss))
            plt.title(myTitle)
            # plt.axis([0, theta_max, 0, 1])
            plt.xlabel('theta')
            plt.ylabel('Curvature')

            # newK1 based
            fig3 = plt.figure(3, figsize=(8, 6))
            ax2 = fig3.add_subplot(111)
            ax2.plot(theta[2:len(theta)-2], newK1[2:len(curvature)-2])
            plt.plot(theta[2:len(theta)-2], newK1_noised_gauss[2:len(newK1_noised_gauss)-2])
            plt.legend(('Original', 'Noised'))
            myTitle = 'Synthetic helix ||U_d(t)|| / ||r_d(t)|| vs time\nSigma=' + str(sigma_noise) + '\nRMS =' + str(computeRMS(newK1, newK1_noised_gauss))
            plt.title(myTitle)
            # plt.axis([0, theta_max, 0, 1])
            plt.xlabel('theta')
            plt.ylabel('newK1')

            # # newK2 based
            # fig4 = plt.figure(4, figsize=(8, 6))
            # ax2 = fig4.add_subplot(111)
            # ax2.plot(theta[2:len(theta)-2], newK2[2:len(curvature)-2])
            # plt.plot(theta[2:len(theta)-2], newK2_noised_gauss[2:len(newK2_noised_gauss)-2])
            # plt.legend(('Original', 'Noised'))
            # myTitle = 'Synthetic helix sqrt(||U_d(t)||^2 + ||L_d(t)||^2) / ||r_d(t)|| vs time\nSigma=' + str(sigma_noise) + '\nRMS =' + str(computeRMS(newK2, newK2_noised_gauss))
            # plt.title(myTitle)
            # # plt.axis([0, theta_max, 0, 1])
            # plt.xlabel('theta')
            # plt.ylabel('newK2')

        # plots all together
        fig5 = plt.figure(5, figsize=(10, 8))
        ax1 = fig5.add_subplot(221, projection='3d')
        # original unnoised data
        ax1.plot(allPoints[:,0], allPoints[:,1], allPoints[:,2], 'b', lw=2)
        # noised data after gaussian smooth
        ax1.plot(allPoints_noised_gauss[:, 0], allPoints_noised_gauss[:, 1], allPoints_noised_gauss[:, 2], 'r', lw=2)
        # noised data after my smooth method
        ax1.plot(allPoints_noised_novel[:, 0], allPoints_noised_novel[:, 1], allPoints_noised_novel[:, 2], 'r', lw=2)
        # plot the pseudo-side vector
        # get the pseudo side vector first
        S = computePseudoSideVector(allPoints)
        origin = [0, 0, 0]
        X, Y, Z = zip(origin)
        U, V, W = zip(S)
        ax1.quiver(X, Y, Z, U, V, W, color='k')
        
        # generate three kinds of kymograph
        # curvature based
        ax2 = fig5.add_subplot(222)
        ax2.plot(theta[2:len(theta)-2], curvature[2:len(curvature)-2])
        plt.plot(theta[2:len(theta)-2], curvature_noised_gauss[2:len(curvature_noised_gauss)-2])
        plt.plot(theta[2:len(theta)-2], curvature_noised_my[2:len(curvature_noised_my)-2])
        plt.legend(('Original', 'Noised after Gauss smooth', 'Noised after my smooth'))
        myTitle = '(||T_d(t)|| / ||r_d(t)||) vs time'
        plt.title(myTitle)
        # plt.axis([0, theta_max, 0, 1])
        plt.xlabel('theta')
        plt.ylabel('Curvature')

        # newK1 based
        ax2 = fig5.add_subplot(223)
        ax2.plot(theta[2:len(theta)-2], newK1[2:len(curvature)-2])
        plt.plot(theta[2:len(theta)-2], newK1_noised_gauss[2:len(newK1_noised_gauss)-2])
        plt.plot(theta[2:len(theta)-2], newK1_noised_my[2:len(newK1_noised_my)-2])
        plt.legend(('Original', 'Noised after Gauss smooth', 'Noised after my smooth'))
        myTitle = '||U_d(t)|| / ||r_d(t)|| vs time'
        plt.title(myTitle)
        # plt.axis([0, theta_max, 0, 1])
        plt.xlabel('theta')
        plt.ylabel('newK1')

        # # newK2 based
        # ax2 = fig5.add_subplot(224)
        # ax2.plot(theta[2:len(theta)-2], newK2[2:len(curvature)-2])
        # plt.plot(theta[2:len(theta)-2], newK2_noised_gauss[2:len(newK2_noised_gauss)-2])
        # plt.plot(theta[2:len(theta)-2], newK2_noised_my[2:len(newK2_noised_my)-2])
        # plt.legend(('Original', 'Noised after Gauss smooth', 'Noised after my smooth'))
        # myTitle = 'sqrt(||U_d(t)||^2 + ||L_d(t)||^2) / ||r_d(t)|| vs time'
        # plt.title(myTitle)
        # # plt.axis([0, theta_max, 0, 1])
        # plt.xlabel('theta')
        # plt.ylabel('newK2')

        # show the images
        plt.show()

    # real data mode
    elif mode == 'real':
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
        x_follower = addGaussianNoise(x_pioneer, 0, sigma_noise)
        y_follower = addGaussianNoise(y_pioneer, 0, sigma_noise)
        z_follower = addGaussianNoise(z_pioneer, 0, sigma_noise)

        # 3D scatter plot of followers
        plot3D(2, time_follower, x_follower, y_follower, z_follower, 'Follower locations vs t')
        

        # get the curvature of both pioneer and follower
        curvature_pioneer = computeCurvature(allPositions_pioneer)
        curvature_follower = computeCurvature(allPositions_pioneer)

        # generate kymograph
        plt.figure(3)
        plt.plot(time_pioneer, curvature_pioneer)
        plt.plot(time_follower, curvature_follower)
        plt.legend(('Pioneer', 'Follower'))
        myTitle = 'Pioneer(real) and follower(synthetic) neurons positions vs time\nSigma=' + str(sigma_noise) + '\nRMS =' + str(computeRMS(curvature_pioneer, curvature_follower))
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
                x_follower = addGaussianNoise(x_pioneer, 0, sigma_noise)
                y_follower = addGaussianNoise(y_pioneer, 0, sigma_noise)
                z_follower = addGaussianNoise(z_pioneer, 0, sigma_noise)

                curvature_follower = computeCurvature(allPositions_follower)
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
