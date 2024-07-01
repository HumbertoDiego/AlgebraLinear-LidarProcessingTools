import imageio
import numpy as np
import matplotlib.pyplot as plt

def svdcompression(img_channel, k_stop=10):
    U, S, Vt = np.linalg.svd(img_channel, full_matrices=False)
    # print(U.shape, S.shape, Vt.shape ) # (1600, 1200) (1200,) (1200, 1200)
    m = len(U)
    n = len(Vt)
    k = len(S)
    #F = np.allclose(redchannel, np.dot(U * S, Vt))
    u = [U[:,i].reshape((m,1)) for i in range(k)]
    v = [Vt[i,:].reshape((1,n)) for i in range(k)]
    # print(u[0].shape, S.shape, v[0].shape ) #(1600, 1) (1200,) (1, 1200)

    reconstructed = np.zeros((m,n))
    for i in range(k_stop):
        reconstructed += S[i]*np.dot(u[i], v[i]) 

    return reconstructed

im = imageio.v3.imread("junina.jpeg")
redchannel = im[:,:,0]
greenchannel = im[:,:,1]
bluechannel = im[:,:,2]

ks = [10,50,100,150]
recs = [np.zeros(im.shape) for i in range(len(ks))]
for idx, k in enumerate(ks): 
    print(k)
    R_rec = svdcompression(redchannel, k_stop= k)
    G_rec = svdcompression(greenchannel, k_stop= k)
    B_rec = svdcompression(bluechannel, k_stop= k)
    rec = np.zeros(im.shape)
    rec[:,:,0] = R_rec
    rec[:,:,1] = G_rec
    rec[:,:,2] = B_rec
    recs[idx] = rec.astype(np.uint8)


fig, axs = plt.subplots(1, 5)
axs[0].imshow(im)
axs[0].set_title("Original")
for idx,k in enumerate(ks):
    axs[idx+1].imshow(recs[idx])
    axs[idx+1].set_title(f"k={k}")
plt.show()