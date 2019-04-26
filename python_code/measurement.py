import cv2
import numpy as np

try:
    img = cv2.imread("../images/img_fil_2.PNG")
    # img = cv2.imread("example_02.png")

except:
    print("Could not load an image, try again.")

# set color tha you are looking for
boundaries = [
    ([0, 0, 0], [128, 128, 128])
]

print(img.shape)
height = img.shape[0]
width = img.shape[1]
img2 = img.copy()

print("width: {} \nheight: {}".format(width, height))

# testing by adding black circle
# cv2.circle(img, (int(width / 2), int(height / 2)), 50, (0, 0, 255), -1)
cv2.rectangle(img, (0, 0), (int(width / 2) - 100, height), (255, 255, 255), -1)
cv2.rectangle(img, (int(width / 2) + 100, 0), (width, height), (255, 255, 255), -1)

width_half = int(width / 2)


cv2.rectangle(img2, (0, 0), (int(width / 2) - 40, height), (255, 255, 255), -1)
cv2.rectangle(img2, (int(width / 2) + 160, 0), (width, height), (255, 255, 255), -1)
# cv2.rectangle(img2, (int(width_half) + 800, height), (width, 0), (255, 255, 255), -1)


for (lower, upper) in boundaries:
    # make numpy array
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    # find colors & make a mask
    mask = cv2.inRange(img, lower, upper)
    output = cv2.bitwise_and(img, img, mask=mask)

    mask2 = cv2.inRange(img2, lower, upper)
    output2 = cv2.bitwise_and(img2, img, mask=mask2)

    # set values
    total_pixels = output.size
    black_pixels = np.count_nonzero(output)
    black_percentage = (black_pixels / total_pixels) * 100

    total_pixels2 = output2.size
    black_pixels2 = np.count_nonzero(output2)
    black_percentage2 = (black_pixels2 / total_pixels2) * 100

    # print values
    print("Total pixels: {}".format(total_pixels))
    print("Black pixels: {}".format(black_pixels))
    print("Percentage: {:.3f}%".format(black_percentage))
    print("-----------------------------------------")
    print("2 -Total pixels: {}".format(total_pixels2))
    print("2 - Black pixels: {}".format(black_pixels2))
    print("2 - Percentage: {:.3f}%".format(black_percentage2))

    percentage_diff = abs(black_percentage - black_percentage2)
    print("-----------------------------------------")
    print("filament difference: {:.3f}%".format(percentage_diff))

cv2.imshow("image1", img)
cv2.imshow("image2", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
