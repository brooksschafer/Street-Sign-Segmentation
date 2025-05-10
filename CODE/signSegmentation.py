from skimage import io, color, morphology, segmentation
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import os

def cleanMask(binaryMask, method):
    binaryMask = morphology.remove_small_objects(binaryMask, min_size=1000)
    binaryMask = morphology.remove_small_holes(binaryMask, area_threshold=2000)
    binaryMask = morphology.closing(binaryMask, morphology.square(3))
    binaryMask = morphology.opening(binaryMask, morphology.square(2))
    cleanedMask = morphology.remove_small_holes(binaryMask, area_threshold=1000)
    if (method != "K-Means (k=2)"):
        cleanedMask = segmentation.clear_border(cleanedMask)

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

def plotResults(img, fileName, binaryMask, cleanedMask, method):
    segmentedImg = img.copy()
    segmentedImg[~cleanedMask] = 0

    name, extension = os.path.splitext(fileName)
    if (method == "Thresholding"):
        fileName = f"{name}_thresholding{extension}"
    elif (method == "K-Means (k=4)"):
        fileName = f"{name}_kMeans{extension}"
    outputPath = os.path.join('OUTPUT', fileName)
    if (method != "K-Means (k=2)"):
        plt.imsave(outputPath, segmentedImg)

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

for fileName in os.listdir('INPUT'):
    imgPath = os.path.join('INPUT', fileName)
    
    #Load and preprocess image
    img = np.asarray(io.imread(imgPath))
    img = color.rgba2rgb(img)

    #Convert to HSV for segmentation
    hsvImg = color.rgb2hsv(img)
    h, s, v = hsvImg[:,:,0], hsvImg[:,:,1], hsvImg[:,:,2]
    plotHSV(h, s, v)

    #Thresholding by saturation value
    threshold = 0.75
    binaryMask = s > threshold

    plt.figure(figsize=(6, 4))
    plt.hist(s.ravel(), bins=256, range=(0, 1), color='blue', alpha=0.7)
    plt.axvline(threshold, color='red', linestyle='-', label=f'Threshold = {threshold}')
    plt.title("Saturation Histogram")
    plt.xlabel("Saturation Value")
    plt.ylabel("# of Pixels")
    plt.legend() 
    plt.tight_layout()
    plt.show()

    cleanedMask = cleanMask(binaryMask, "Thresholding")
    plotResults(img, fileName, binaryMask, cleanedMask, "Thresholding")

    #K-means clustering (k=2)
    newImg = hsvImg.reshape(-1, 3)
    kMeans = KMeans(n_clusters=2, n_init=10)
    kMeans.fit(newImg)
    binaryMask = (kMeans.labels_ == 1).reshape(hsvImg.shape[0], hsvImg.shape[1])

    cleanedMask = cleanMask(binaryMask, "K-Means (k=2)")
    plotResults(img, fileName, binaryMask, cleanedMask, "K-Means (k=2)")

    #K-means clustering (k=4)
    kMeans = KMeans(n_clusters=4, n_init=10)
    kMeans.fit(newImg)
    labels = kMeans.labels_
    clusteredImg = labels.reshape(hsvImg.shape[0], hsvImg.shape[1])

    plt.imshow(clusteredImg, cmap='viridis')
    plt.title("K-Means Clusters (k=4)")
    plt.tight_layout()
    plt.show()

    #Get smallest cluster for mask
    uniqueLabels, counts = np.unique(labels, return_counts=True)
    smallestCluster = uniqueLabels[np.argmin(counts)]
    binaryMask = (labels == smallestCluster)
    binaryMask = binaryMask.reshape(hsvImg.shape[0], hsvImg.shape[1])

    cleanedMask = cleanMask(binaryMask, "K-Means (k=4)")
    plotResults(img, fileName, binaryMask, cleanedMask, "K-Means (k=4)")

print("Program complete")
