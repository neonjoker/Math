import numpy as np
from matplotlib import pyplot as plt
import PIL


def svdimage(filename, percent):
    original = plt.imread(filename)
    R0 = np.array(original[:, :, 0])
    G0 = np.array(original[:, :, 1])
    B0 = np.array(original[:, :, 2])
    u0, sigma0, v0 = np.linalg.svd(R0)
    u1, sigma1, v1 = np.linalg.svd(G0)
    u2, sigma2, v2 = np.linalg.svd(B0)
    R1 = np.zeros(R0.shape)
    G1 = np.zeros(G0.shape)
    B1 = np.zeros(B0.shape)
    for i in range(int(percent * len(sigma0)) + 1):
        R1 += sigma0[i] * np.dot(u0[:, i].reshape(-1, 1), v0[i, :].reshape(1, -1))
    for i in range(int(percent * len(sigma1)) + 1):
        G1 += sigma1[i] * np.dot(u1[:, i].reshape(-1, 1), v1[i, :].reshape(1, -1))
    for i in range(int(percent * len(sigma2)) + 1):
        B1 += sigma2[i] * np.dot(u2[:, i].reshape(-1, 1), v2[i, :].reshape(1, -1))
    final = np.stack((R1, G1, B1), 2)
    final[final > 255] = 255
    final[final < 0] = 0
    final = np.rint(final).astype('uint8')
    return final


if __name__ == '__main__':
    filename = 'TestPicture.jpg'
    for p in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        after = svdimage(filename, p)
        plt.imsave(str(p) + '_0.jpg', after)
