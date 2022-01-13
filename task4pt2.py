import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#influenced by https://code.likeagirl.io/finding-dominant-colour-on-an-image-b4e075f98097
cap = cv2.VideoCapture(0)





def find_histogram(clt):
    """
    create a histogram with k clusters
    :param: clt
    :return:hist
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist

def plot_colors2(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar



while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame2 = frame2.reshape((frame2.shape[0] * frame2.shape[1],3))
    clt = KMeans(n_clusters=3)  # cluster number
    clt.fit(frame2)

    hist = find_histogram(clt)
    bar = plot_colors2(hist, clt.cluster_centers_)

    cv2.imshow('frame', frame)
    plt.axis("off")
    plt.imshow(bar)
    plt.show()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()



