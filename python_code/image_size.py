
# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
# import argparse
import imutils
import cv2

# calculate midpoints between img lines
def midpoint(ptA, ptB):
    return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5


img_input = cv2.imread("/Users/petrzamecnik/PycharmProjects/opencv/images/img_fil_2.PNG")
input_img_width = int(img_input.shape[0])
input_img_height = int(img_input.shape[1])

width_20 = (input_img_width / 100) * 20

cv2.rectangle(img_input, (0, 0), (int(input_img_width / 2), int(input_img_height)), (255, 255, 255), -1)
cv2.rectangle(img_input, (int(input_img_width / 2 + (width_20 * 1.8)), 0), (int(input_img_width * 2), int(input_img_height)),
              (255, 255, 255), -1)
cv2.circle(img_input, (55, 55), 50, (0, 0, 255), -1)
cv2.circle(img_input, (160, 55), 50, (5, 5, 5), -1)
cv2.imshow("img", img_input)

boundaries = [
    ([0, 0, 0], [128, 128, 128])  #  look for pixels between black and gray
]

img_name = "my_image"

for (lower, upper) in boundaries:
    # creates numpy array from boundaries
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    # finds colors in boundaries a applies a mask
    mask = cv2.inRange(img_input, lower, upper)
    output = cv2.bitwise_and(img_input, img_input, mask=mask)

    tot_pixel = output.size
    black_pixel = np.count_nonzero(output)
    percentage = round(black_pixel * 100 / tot_pixel, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_input, ("total pixels: {}".format(tot_pixel)), (15, 300), font, 0.5, (64, 64, 64), 2, cv2.LINE_AA)
    cv2.putText(img_input, ("black pixels: {}".format(black_pixel)), (15, 320), font, 0.5, (64, 64, 64), 2, cv2.LINE_AA)
    cv2.putText(img_input, ("black px percentage: {}%".format(percentage)), (15, 340), font, 0.5, (64, 64, 64), 2,
                cv2.LINE_AA)

    print("Black pixels: " + str(black_pixel))
    print("Total pixels: " + str(tot_pixel))
    print("Percentage of black pixels: " + str(percentage) + "%")
    print("-----------------------------------------------------")

# img_gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
# black = img_input[0, 0, 0]
# white = img_input[100, 100, 0]
# print("black: ", black)
# print("white: ", white)
#
# print("img shape: ", img_input.shape)
# print("img size: ", img_input.size)
# print("img width: ", input_img_width)
# print("img height: ", input_img_height)
# print("ehm?: ", int(input_img_width * input_img_height))

# cv2.circle(resized_image, (60, 60), 50, (0, 0, 255), -1)
# cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 00), 2)

# load the image, convert it to grayscale, and blur it slightly
# image = cv2.imread("/Users/petrzamecnik/PycharmProjects/opencv/images/img2.jpeg")
image = img_input
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)

# scale_percent = 100  # percent of original size
# width = int(image.shape[0] * scale_percent / 100)
# height = int(image.shape[1] * scale_percent / 100)
# dim = (width, height)

# width = float(image.shape[0])
# height = float(image.shape[1])

width = 1.75
height = 1.75

# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
# edged = cv2.Canny(gray, 20, 30)  # works great for coints, cant see filament
edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

# find contours in the edge map
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# sort the contours from left-to-right and initialize the
# 'pixels per metric' calibration variable
(cnts, _) = contours.sort_contours(cnts)
pixelsPerMetric = None

# loop over the contours individually
for c in cnts:
    # if the contour is not sufficiently large, ignore it
    if cv2.contourArea(c) < 100:
        continue

    # compute the rotated bounding box of the contour
    orig = image.copy()
    box = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")

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
    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)

    # compute the midpoint between the top-left and top-right points,
    # followed by the midpoint between the top-righ and bottom-right
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

    # if the pixels per metric has not been initialized, then
    # compute it as the ratio of pixels to supplied metric
    # (in this case, inches)
    if pixelsPerMetric is None:
        pixelsPerMetric = dB / width

    # compute the size of the object
    dimA = dA / pixelsPerMetric
    dimB = dB / pixelsPerMetric

    # dimAcm = (dimA / 2.54) / 2
    # dimBcm = (dimB / 2.54) / 2

    # draw the object sizes on the image
    cv2.putText(orig, "{:.3f}mm".format(dimA),
                (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 0, 0), 2)
    cv2.putText(orig, "{:.3f}mm".format(dimB),
                (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 0, 0), 2)

    # show the output image
    cv2.imshow("Image", orig)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
