# Lab Assessment 02 - Morphological operation on sample image - Dilation and Erosion

import cv2
import numpy as np

# 1. Read the image
img = cv2.imread('6307733789072887313.png')   # replace with your image path

if img is None:
    print("Error: Image not found!")
    exit()

# 2. Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3. Convert to binary image using thresholding
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 4. Create structuring element (kernel)
kernel = np.ones((5, 5), np.uint8)

# 5. Apply morphological operations

# Dilation
dilation = cv2.dilate(binary, kernel, iterations=1)

# Erosion
erosion = cv2.erode(binary, kernel, iterations=1)

# Opening (Erosion followed by Dilation)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

# Closing (Dilation followed by Erosion)
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# 6. Display results
cv2.imshow("Original Image", img)
cv2.imshow("Grayscale", gray)
cv2.imshow("Binary", binary)
cv2.imshow("Dilation", dilation)
cv2.imshow("Erosion", erosion)
cv2.imshow("Opening", opening)
cv2.imshow("Closing", closing)

# 7. Save results (optional)
cv2.imwrite("binary.jpg", binary)
cv2.imwrite("dilation.jpg", dilation)
cv2.imwrite("erosion.jpg", erosion)
cv2.imwrite("opening.jpg", opening)
cv2.imwrite("closing.jpg", closing)

# 8. Wait and close windows
cv2.waitKey(0)
cv2.destroyAllWindows()
