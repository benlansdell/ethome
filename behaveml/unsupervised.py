from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import numpy as np
from scipy import signal

def compute_tsne_embedding(data, cols, N_rows = 20000):
    tsne_data = StandardScaler().fit_transform(data[cols])
    random_indices = np.random.choice(tsne_data.shape[0], N_rows, replace = False)
    tsne_data = tsne_data[random_indices, :]
    tsne_embedding = TSNE(n_components=2, init = 'pca').fit_transform(tsne_data)
    return tsne_embedding, random_indices

def morlet(data, dt, w = 6, n_freq = 25):
    fs = 1/dt
    freq = np.geomspace(1, fs/2, n_freq)
    widths = w*fs / (2*freq*np.pi)
    cwtm = signal.cwt(data, signal.morlet2, widths, w=w)
    return cwtm

def watershed(density_matrix):
    # import the necessary packages
    from skimage.feature import peak_local_max
    from skimage.morphology import watershed
    from scipy import ndimage
    import numpy as np
    import argparse
    import imutils
    import cv2

    # construct the argument parse and parse the arguments
    # load the image and perform pyramid mean shift filtering
    # to aid the thresholding step
    im_path = ''

    image = cv2.imread(im_path)
    shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)
    cv2.imshow("Input", image)
    # convert the mean shift image to grayscale, then apply
    # Otsu's thresholding
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imshow("Thresh", thresh)

    # compute the exact Euclidean distance from every binary
    # pixel to the nearest zero pixel, then find peaks in this
    # distance map
    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=20,
        labels=thresh)
    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)
    print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))

    # loop over the unique labels returned by the Watershed
    # algorithm
    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255
        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        # draw a circle enclosing the object
        ((x, y), r) = cv2.minEnclosingCircle(c)
        cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv2.putText(image, "#{}".format(label), (int(x) - 10, int(y)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    # show the output image
    cv2.imshow("Output", image)