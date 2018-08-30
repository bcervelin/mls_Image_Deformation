# affine image deformation using MLS
# this code was implemented by B.H.Cervelin - 2018
# you can use it as you wish, but we will be very pleased if you cite our
# presentation on Brasilian National Congress on Applied and Computational
# Mathematics (CNMAC)
# http://www.sbmac.org.br/eventos/cnmac/xxxiii_cnmac/pdf/604.pdf
# The presentation was inspired on the original paper:
# S. Schaefer, T. McPhail, J. Warren, Image deformation using moving least
# squares. ACM SIGGRAPH, pp. 533-540, 2006.

import numpy as np
import cv2
from scipy import interpolate as ip

# deformation internal parameters
alpha = 2.

# read image name and its extension
nome = input("Enter image name without it's extension: ")
tipo = input("Enter image extension: ")
A = cv2.imread(nome + '.' + tipo, 1)

# read data from terminal
a = 1
print("\n All the pixels coordinates must be separated by comma ','")
p = []
q = []
while (a != 0):
    q1 = eval('[' + input(
            "type the position of moved pixel at original image \n"
            ) + ']')
    p1 = eval('[' + input(
            "type the expected position pixel at deformed image \n"
            ) + ']')
    p.append(p1)
    q.append(q1)
    a = eval(input(
        "type '0' to stop adding points or anything else to continue\n"))
p_list = p
p = np.array(p)
q = np.array(q)
k = len(p)

# find deformation
# dimensions of image A
m, n = A.shape[:2]
T = np.maximum(m, n)
AA = np.zeros([T, T, 3])
AA[:m, :n, :] = A
A = AA
d1, d2 = int((T)/10)+1, int((T)/10)+1
# build positions for deformation
Xa = np.zeros([int((T)/d1+1), int((T)/d2+1)])
Ya = Xa
for i in range(0, T, d1):
    ii = int((i)/d1)
    for j in range(0, T, d2):
        jj = int((j)/d2)
        # verify if v is in p
        try:
            pos = p_list.index([i, j])
            # if it is, force deformation
            Xa[ii, jj], Ya[ii, jj] = q[pos, 0], q[pos, 1]
        # if it isnt use MLS
        except ValueError:
            v = np.array([i, j])
            # evaluate weights of each point
            w = 1./np.power(np.linalg.norm(p - v, axis=1), (2.*alpha))
            w_sum = np.sum(w)
            # find weighted centroids
            p1 = np.sum(np.transpose([w, w])*p, axis=0)/w_sum
            q1 = np.sum(np.transpose([w, w])*q, axis=0)/w_sum
            # find translation
            p2, q2 = p - p1, q - q1
            # affine deformation
            fa = np.zeros(2)
            mu = np.zeros([2, 2])
            for l in range(k):
                mu += w[l]*np.outer(p2[l, :], p2[l, :])
            mu = np.linalg.inv(mu)
            for l in range(k):
                M = w[l]*np.dot(v-p1, np.matmul(mu, p2[l, :]))
                fa += M*q2[l, :]
            fa += q1
            # write deformation
            Xa[ii, jj], Ya[ii, jj] = fa
# interpolate deformation
ipX = ip.RectBivariateSpline(range(0, T, d1), range(0, T, d2), Xa)
XX = ipX(range(T), range(T))
ipY = ip.RectBivariateSpline(range(0, T, d1), range(0, T, d2), Ya)
YY = ipX(range(T), range(T))
YY = np.minimum(np.maximum(YY, 0), n-1)
XX = np.transpose(XX.astype(int))
XX = np.minimum(np.maximum(XX, 0), m-1)
# implement deformation
ipAr = ip.RectBivariateSpline(range(T), range(T), A[:, :, 0])
ipAb = ip.RectBivariateSpline(range(T), range(T), A[:, :, 1])
ipAg = ip.RectBivariateSpline(range(T), range(T), A[:, :, 2])
B = np.zeros(A.shape)
for i in range(T):
    for j in range(T):
        B[i, j, 0] = ipAr(XX[i, j], YY[i, j])
        B[i, j, 1] = ipAb(XX[i, j], YY[i, j])
        B[i, j, 2] = ipAg(XX[i, j], YY[i, j])

cv2.imwrite(nome + '_affine.' + tipo, B[:m, :n, :])
