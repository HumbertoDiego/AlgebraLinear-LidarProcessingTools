"""
Implementação da compressão de uma matriz por truncamento da decomposição SVD
"""
import sys
import imageio
import numpy as np
import torch
import matplotlib.pyplot as plt

def svdcompression(img_channel, k_stop=10):
    U, S, Vt = torch.linalg.svd(img_channel, full_matrices=False)
    print(U.shape, S.shape, Vt.shape ) # [675, 675] [675]) [675, 1200]
    reconstructed = U[:, :k_stop] @ torch.diag(S[:k_stop]) @ Vt[:k_stop, :]
    return reconstructed.numpy()

# Teste um uma imagem

if __name__== "__main__":
    filename = "data/junina.jpg"
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    im = imageio.v3.imread(filename)
    redchannel = torch.tensor(im[:,:,0], dtype=torch.uint8).float()
    greenchannel = torch.tensor(im[:,:,1], dtype=torch.uint8).float()
    bluechannel = torch.tensor(im[:,:,2], dtype=torch.uint8).float()
    print(im.shape)

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
    for idx, k in enumerate(ks):
        axs[idx+1].imshow(recs[idx])
        axs[idx+1].set_title(f"k={k}")
    plt.show()