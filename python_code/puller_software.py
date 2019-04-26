import cv2
import numpy as np

# set basic variables
location = "../images/img_fil_2.PNG"
boundaries = [
    ([0, 0, 0], [128, 128, 128])  # looking for those colors
]
height = 0
width = 0


def get_img_data():
    height = img.shape[0]
    width = img.shape[1]
    return height, width


def draw_recntangle(width, height):
    cv2.rectangle(img, (0, 0), (int(width / 2) - 60, height), (255, 255, 255), -1)
    cv2.rectangle(img, (int(width / 2) + 60, 0), (width, height), (255, 255, 255), -1)


def get_values_of_black():
    for (lower, upper) in boundaries:
        # make numpy array
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        # find colors & make a mask
        mask = cv2.inRange(img, lower, upper)
        output = cv2.bitwise_and(img, img, mask=mask)

        # set values
        total_pixels = output.size
        black_pixels = np.count_nonzero(output)
        black_percentage = (black_pixels / total_pixels) * 100

        return total_pixels, black_pixels, black_percentage


# load image
img = cv2.imread(location)

# main setup
sizes = get_img_data()  # 1 is width, 0 is height   # Size of image
draw_recntangle(sizes[1], sizes[0])
values = get_values_of_black()

# print values
print("height is: {}\nwidth is:  {}".format(sizes[0], sizes[1]))
print("Total pixels: {}".format(values[0]))
print("Black pixels: {}".format(values[1]))
print("Percentage of black : {:.3f}%".format(values[2]))

cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()