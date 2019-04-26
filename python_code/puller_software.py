import cv2
import numpy as np




# set basic variables
location = "../images/img_fil_2.PNG"
# location = "../images/filament_mirrors2.jpg"
boundaries = [([0, 0, 0], [80, 80, 80])]  # looking for those colors
gray_boundaries = [(0, 120)]
image_width = 3265
image_height = 2448
image_width_half = int(image_width / 2)
image_height_half = int(image_height / 2)
# img_0_crop_values = int[(1070, 1570), (1070, 1695), (970, 1695), (970, 1570)]
img_0_crop_values_y = [1070, 1070, 970, 970]
img_0_crop_values_x = [1570, 1695, 1695, 170]
# [(1070, 1570), (1070, 1695), (970, 1695), (970, 1570)]
# [y0x0, y0x1, y1x0, y1x1]


def get_img_data():
    height = img.shape[0]
    width = img.shape[1]
    return height, width


def draw_rectangle(width, height):
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
# img_top = img[960: image_height_half + 50, 1450: (image_width + 100) - (image_width_half - 100)]

# img_0 = img[(img_0_crop_values_y[0], img_0_crop_values_x[0]): (img_0_crop_values_y[1], img_0_crop_values_x[1]),
#         (img_0_crop_values_y[2], img_0_crop_values_x[2]): (img_0_crop_values_y[3], img_0_crop_values_x[3])]
# img_1 = img

# crop_img = img[y:y+h, x:x+w]
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# main setup
sizes = get_img_data()  # 1 is width, 0 is height   # Size of image
draw_rectangle(sizes[1], sizes[0])
values = get_values_of_black()
# values_gray = get_values_of_black_grayscale()

# print values
print("height is: {}px\nwidth is:  {}px".format(sizes[0], sizes[1]))
print("Total pixels: {}".format(values[0]))
print("Black pixels: {}".format(values[1]))
print("Percentage of black : {:.3f}%".format(values[2]))

# show image, destroy windows
cv2.imshow("image", img)
cv2.setWindowTitle("image", "img")
cv2.namedWindow("image", flags=cv2.WINDOW_GUI_EXPANDED)
cv2.setWindowProperty("image", cv2.WINDOW_GUI_EXPANDED, cv2.WINDOW_GUI_EXPANDED)
# cv2.imshow("img_1", img_1)

# cv2.imshow("image_gray", gray_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
