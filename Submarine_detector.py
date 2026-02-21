"""The data set we need is : 
An absolute ton of normal ocean images. i.e Free from a submarine or any other object we wish to detect. Intstead
of identifying the presence of an abnormality, I will detenct the absence of normality. 
With a tight enough definition of normality, we can detect the presence of an object by the absence of normality."""

"""I will use a pretrained CNN over a self supervised autoencoder , quite simply because currently it outpoerfroms it ."""

import os
from glob import glob
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from math import floor


## DATA_DIR = "/path/to/normal_ocean_images"  # only normal images here

class OceanDataset(Dataset):
    def __init__(self, root, transform=None): #I dont see any reason to use the transform here, but I will keep it for now
        self.paths = sorted(
            glob(os.path.join(root, "**", "*.*"), recursive=True) #this will get all the images in the directory and subdirectories
        )
        self.transform = transform #this is for the pretrained CNN, we need to resize and normalize the images

    def __len__(self):
        return len(self.paths) #no. of images 

    def __getitem__(self, idx): #idx is the index of the image we want to get
        path = self.paths[idx]  #making life easier 
        img = Image.open(path).convert("RGB") #coverting to RGB 
        if self.transform: 
            img = self.transform(img) #no trasformations for now, but we will need to resize and normalize the images for the pretrained CNN
        return img, path

transform = T.Compose([ # T is for transforms, we will use it to resize and normalize the images for the pretrained CNN
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], #these means were chosen because they are the mean and std of the ImageNet dataset, which is what the pretrained CNN was trained on
                std=[0.229, 0.224, 0.225]),
])

dataset = OceanDataset(DATA_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #We will use the GPU if it is available, otherwise we will use the CPU.

"""Now we will use the pretrained CNN to extract features from the images. We will use the ResNet50 model."""
backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
backbone.fc = nn.Identity()   # 2048-dim features 
backbone.eval().to(device) #if we are training on a device, I usually use the cloud.  


def extract_features(dataloader, model, device):
    feats = []
    paths = []
    for imgs, img_paths in dataloader:
        imgs = imgs.to(device)
        f = model(imgs)              # ResNet will return a tensor with the shape (B, 2048), B is the batch size, 2048 is the feature dimension
        f = f.cpu().numpy()
        feats.append(f)
        paths.extend(img_paths)
    feats = np.concatenate(feats, axis=0)
    return feats, paths

features, img_paths = extract_features(loader, backbone, device)
print(features.shape)  # (N, 2048)


"""I will use PCA to reduce the dimensionality of the features to 128. Efficiency rules!"""
pca_dim = 128
pca = PCA(n_components=pca_dim, random_state=42)
features_pca = pca.fit_transform(features)
print(features_pca.shape)  # (N, 128)

"""Now we will use KMeans to cluster the features. We will use 10 clusters for now, but we can tune this later."""

k = 10  # tune this
kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
kmeans.fit(features_pca)

centers = kmeans.cluster_centers_          # (k, d)
labels = kmeans.labels_                    # (N,)

train_dists = np.linalg.norm( # The distance from its own centroid. 
    features_pca - centers[labels],
    axis=1
)
"""I will now define a load and embed function that will take an image path, load the image, extract its features using the pretrained CNN, and then reduce the dimensionality using PCA. This will be used for anomaly detection on new images."""
def load_and_embed(path):
    # 1. Load and preprocess image
    img = Image.open(path).convert("RGB")
    img = transform(img)          # resize + normalize
    img = img.unsqueeze(0).to(device)  # (1, 3, H, W)

    # 2. Extract CNN features
    with torch.no_grad():
        f = backbone(img)         # (1, 2048) for ResNet50
    f = f.cpu().numpy()[0]        # (2048,)

    # 3. Project into PCA space (same as training)
    z = pca.transform(f[None, :])[0]   # (pca_dim,)

    return z
"""Now we will use the distances to determine a threshold for anomaly detection. We will use the 99th percentile of the distances as the threshold, but we can tune this later."""
percentile = 99
threshold = np.percentile(train_dists, percentile)
print("Distance threshold:", threshold)
def kmeans_anomaly_score(path):
    z = load_and_embed(path)          # raw feature, (2048,)
    z_pca = pca.transform(z[None, :])[0]  # (d,)
    dists = np.linalg.norm(centers - z_pca[None, :], axis=1)
    score = dists.min()
    return score, score > threshold

test_img = "/path/to/test_image.png"
score, is_anom = kmeans_anomaly_score(test_img)
print(f"Score = {score:.3f}, anomaly? {is_anom}")

"""I will now use patch level features to create a heatmap."""
patch_size = 64   # pixels
stride = 64       # same as patch_size → non-overlapping

def extract_patches(img_tensor, patch_size=64, stride=64):
    """
    img_tensor: (1, 3, H, W), already transformed
    returns: patches (P, 3, patch_size, patch_size)
    """
    _, _, H, W = img_tensor.shape
    patches = []

    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = img_tensor[:, :, y:y+patch_size, x:x+patch_size]  # (1, 3, ph, pw)
            patches.append(patch)

    patches = torch.cat(patches, dim=0)  # (P, 3, patch_size, patch_size)
    return patches


def embed_patches(patches):
    """
    patches: (P, 3, H, W)
    returns: z_pca: (P, d)
    """
    patches = patches.to(device)
    feats = backbone(patches)          # (P, 2048)
    feats = feats.cpu().numpy()
    z_pca = pca.transform(feats)       # (P, d)
    return z_pca

"""We will now do k means for each patch and create a heatmap of the anomaly scores."""
def patch_anomaly_scores(z_pca, centers):
    """
    z_pca: (P, d), centers: (k, d)
    returns: scores: (P,)
    """
    # broadcasting: (P, 1, d) - (1, k, d) -> (P, k, d)
    diff = z_pca[:, None, :] - centers[None, :, :]   # (P, k, d)
    dists = np.linalg.norm(diff, axis=2)             # (P, k)
    scores = dists.min(axis=1)                       # (P,)
    return scores

def scores_to_heatmap(scores, img_tensor, patch_size=64, stride=64):
    """
    scores: (P,)
    returns: heatmap: (H_p, W_p) where H_p, W_p = number of patches along H and W
    """
    _, _, H, W = img_tensor.shape
    H_p = (H - patch_size) // stride + 1
    W_p = (W - patch_size) // stride + 1
    heatmap = scores.reshape(H_p, W_p)
    return heatmap
"""Finally, we will put everything together in a function that takes an image path, extracts patch features, computes anomaly scores, and returns a heatmap."""
def patch_level_anomaly(image_path):
    # load + transform
    img = Image.open(image_path).convert("RGB")
    img_t = transform(img).unsqueeze(0)  # (1, 3, H, W)

    # patches
    patches = extract_patches(img_t, patch_size=patch_size, stride=stride)  # (P, 3, ph, pw)

    # embeddings
    z_pca = embed_patches(patches)  # (P, d)

    # K-means anomaly scores
    scores = patch_anomaly_scores(z_pca, centers)  # (P,)

    # heatmap
    heatmap = scores_to_heatmap(scores, img_t, patch_size=patch_size, stride=stride)

    return img, heatmap, scores

test_path = "/path/to/test_image.png"
img, heatmap, scores = patch_level_anomaly(test_path)
print("Max patch anomaly score:", scores.max())



"""# visualize heatmap"""

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.axis("off")
plt.title("Input image")

plt.subplot(1, 2, 2)
plt.imshow(heatmap, cmap="hot")
plt.colorbar(label="Anomaly score")
plt.title("Patch-level anomaly")
plt.axis("off")

plt.tight_layout()
plt.show()
