import cv2
import numpy as np
import imutils
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours



def midpoint(ptA, ptB):
    return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5


# image = cv2.imread("/Users/petrzamecnik/PycharmProjects/opencv/images/image.jpeg")
image = cv2.imread("/Users/petrzamecnik/PycharmProjects/opencv/images/example_01.png")

scale_percent = 100  # percent of original size
width = int(image.shape[0] * scale_percent / 100)
height = int(image.shape[1] * scale_percent / 100)
dim = (width, height)

# resize image
resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
cv2.circle(resized_image, (60, 60), 50, (0, 0, 255), -1)

gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)  # convert to gray scale
gray = cv2.GaussianBlur(gray, (7, 7), 0)  # apply gaussian blur

# edged = cv2.Canny(gray, 50, 100)
# edged = cv2.Canny(gray, 0, 100)

# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edged = cv2.Canny(gray, 80, 160)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

# find contours in edge map
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# sort the contours from left-to-right and initialize the
# 'pixels per metric' calibration variable
(cnts, _) = contours.sort_contours(cnts)
pixelsPerMetric = None

# loop for contours idividually
for c in cnts:
    if cv2.contourArea(c) < 100:  # if the contour is not sufficiently large, ignore it
        continue

    # compute the rotated bounding box of the contour

    orig = image.copy()
    # resized_orig = cv2.resize(orig, dim, interpolation=cv2.INTER_AREA)

    box = cv2.minAreaRect(c)
    box = cv2.boxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, int)

    # order the points in the contour such that they appear
    # in top-left, top-right, bottom-right, and bottom-left
    # order, then draw the outline of the rotated bounding
    # box
    box = perspective.order_points(box)
    cv2.drawContours(orig, [box.astype(int)], -1, (0, 255, 0), 2)

    # loop over the original points and draw them
    for (x, y) in box:
        cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

    # unpack the ordered bounding box, then compute the midpoint
    # between the top-left and top-right coordinates, followed by
    # the midpoint between bottom-left and bottom-right coordinates
    (tl, tr, bl, br) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)

    # compute the midpoint between the top-left and top-right points,
    # followed by the midpoint between the top-right and bottom-right
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    # draw the midpoints on the image
    cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

    # draw lines between the midpoints
    cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
             (255, 0, 255), 2)
    cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
             (255, 0, 255), 2)

    # compute the Euclidean distance between the midpoints

    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    if pixelsPerMetric is None:
        pixelsPerMetric = dB / width

    pixelsPerMetric = dB / width

    # compute the size of the object
    # dimA = dA / pixelsPerMetric
    # dimB = dB / pixelsPerMetric
    #
    # # draw the object sizes on the image
    # cv2.putText(orig, "{:.1f}in".format(dimA),
    #             (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
    #             0.65, (255, 255, 255), 2)
    # cv2.putText(orig, "{:.1f}in".format(dimB),
    #             (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
    #             0.65, (255, 255, 255), 2)

    # show the output image
    # cv2.imshow("Image", orig)
    # cv2.waitKey(0)
    print(type(dA))
    print(dA)

    print(type(width))
    print(width)

    print(type(pixelsPerMetric))
    print(pixelsPerMetric)


# cv2.imshow("image", resized_image)
# cv2.imshow("gray", gray)
# cv2.imshow("edged", edged)
# cv2.imshow("original", orig)
# cv2.imshow("cnts", cnts)  # doesnt work

print('Original Dimensions : ', image.shape)
print("Resized Dimensions : ", resized_image.shape)


cv2.waitKey(0)
cv2.destroyAllWindows()
