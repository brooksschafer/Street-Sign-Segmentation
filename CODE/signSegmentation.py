from skimage import io, color, morphology
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

def cleanMask(binaryMask):
    cleanedMask = morphology.remove_small_objects(binaryMask, min_size=100)
    cleanedMask = morphology.remove_small_holes(cleanedMask, area_threshold=2000)
    cleanedMask = morphology.closing(cleanedMask, morphology.square(3))
    cleanedMask = morphology.opening(cleanedMask, morphology.square(2))
    cleanedMask = morphology.remove_small_holes(cleanedMask, area_threshold=1000)

    return cleanedMask

def plotHSV(h, s, v):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("HSV Values", fontsize=18)

    axs[0].imshow(h, cmap='hsv')
    axs[0].set_title("Hue Channel")
    axs[1].imshow(s, cmap='gray')
    axs[1].set_title("Saturation Channel")
    axs[2].imshow(v, cmap='gray')
    axs[2].set_title("Value Channel")

    plt.tight_layout()
    plt.show()

def plotResults(img, binaryMask, cleanedMask, method):
    segmentedImg = img.copy()
    segmentedImg[~cleanedMask] = 0

    fig, axs = plt.subplots(1, 4, figsize=(15, 4))
    fig.suptitle(method + " Segmentation Results", fontsize=18)

    axs[0].imshow(img)
    axs[0].set_title("Original Image")
    axs[1].imshow(binaryMask, cmap='gray')
    axs[1].set_title("Original Mask")
    axs[2].imshow(cleanedMask, cmap='gray')
    axs[2].set_title("Cleaned Mask")
    axs[3].imshow(segmentedImg)
    axs[3].set_title("Segmented Image")

    plt.tight_layout()
    plt.show()

#Load and preprocess image
img = np.asarray(io.imread("SignSegmentation/INPUT/img6.png"))
img = color.rgba2rgb(img)

#Convert to HSV for segmentation
hsvImg = color.rgb2hsv(img)
h, s, v = hsvImg[:,:,0], hsvImg[:,:,1], hsvImg[:,:,2]
plotHSV(h, s, v)

#Thresholding by saturation value
binaryMask = s > 0.6
cleanedMask = cleanMask(binaryMask)
plotResults(img, binaryMask, cleanedMask, "Thresholding")

#K-means clustering (k=2)
newImg = hsvImg.reshape(-1, 3)
kMeans = KMeans(n_clusters=2, n_init=10)
kMeans.fit(newImg)
binaryMask = (kMeans.labels_ == 1).reshape(hsvImg.shape[0], hsvImg.shape[1])

cleanedMask = cleanMask(binaryMask)
plotResults(img, binaryMask, cleanedMask, "K-Means (k=2)")

print("Program completed")
