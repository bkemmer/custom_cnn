from skimage.feature import local_binary_pattern
import matplotlib,pyplot as plt
# import cv2

def aplica_lbp(path, numPoints=24, radius=8, method='uniform'):
    img = plt.imread(path)
    lbp = local_binary_pattern(img, numPoints, radius, method)
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, numPoints + 3), range=(0, numPoints + 2))
    eps = 1e-6
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)

    return lbp, hist


    