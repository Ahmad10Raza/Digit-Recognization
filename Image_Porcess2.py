import cv2
import numpy as np        
import matplotlib.pyplot as plt
from scipy import ndimage
import math
from keras.models import load_model


# loading pre trained model
model = load_model('Model/Model_ckpt.h5')

def predict_digit(img):
    test_image = img.reshape(-1,28,28,1)
    return np.argmax(model.predict(test_image))


#pitting label
# The function first creates a rectangle around the label using the cv2.rectangle() function.
# The rectangle is green in color and has a thickness of 1 pixel.
def put_label(t_img, label, x, y):

    # Create a font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Calculate the coordinates of the label
    l_x = int(x) - 10
    l_y = int(y) + 10

    # Draw a rectangle around the label
    cv2.rectangle(t_img, (l_x, l_y + 5), (l_x + 35, l_y - 35), (0, 255, 0), -1)

    # Add the label to the image
    cv2.putText(t_img, str(label), (l_x, l_y), font, 1.5, (255, 0, 0), 1, cv2.LINE_AA)

    # Return the image with the label added
    return t_img


# refining each digit
def image_refiner(gray):
    org_size=22
    img_size=28

    # Get the dimensions of the image
    rows, cols = gray.shape

    # Check if the width is greater than the height
    if rows > cols:

        # Calculate the resizing factor
        factor = org_size / rows

        # Resize the image
        rows = org_size
        cols = int(round(cols * factor))

    else:

        # Calculate the resizing factor
        factor = org_size / cols

        # Resize the image
        cols = org_size
        rows = int(round(rows * factor))

    # Resize the image
    gray = cv2.resize(gray, (cols, rows))

    # Get the padding
    colsPadding = (int(math.ceil((img_size - cols) / 2.0)), int(math.floor((img_size - cols) / 2.0)))
    rowsPadding = (int(math.ceil((img_size - rows) / 2.0)), int(math.floor((img_size - rows) / 2.0)))

    # Apply padding
    gray = np.lib.pad(gray, (rowsPadding, colsPadding), 'constant')

    # Return the refined image
    return gray





def get_output_image(path):

    # Load the image and convert it to grayscale
    img = cv2.imread(path, 2)
    img_org = cv2.imread(path)

    # Apply a threshold to the image to binarize it
    ret, thresh = cv2.threshold(gray, 127, 255, 0)

    # Find all of the contours in the image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # For each contour, check if it is a valid digit
    for j, cnt in enumerate(contours):

        # Calculate the epsilon value for approximation
        epsilon = 0.01 * cv2.arcLength(cnt, True)

        # Approximate the contour
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Calculate the convex hull of the contour
        hull = cv2.convexHull(cnt)

        # Check if the contour is convex
        k = cv2.isContourConvex(cnt)

        # Get the bounding rectangle of the contour
        x, y, w, h = cv2.boundingRect(cnt)

        # Check if the contour is a valid digit
        if (hierarchy[0][j][3] != -1 and w > 10 and h > 10):

            # Put a boundary around the digit
            cv2.rectangle(img_org, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Crop the digit from the image
            roi = img[y:y + h, x:x + w]

            # Apply a bitwise not operation to the cropped image
            roi = cv2.bitwise_not(roi)

            # Refine the cropped image
            roi = image_refiner(roi)

            # Apply a threshold to the refined image
            th, fnl = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY)

            # Predict the digit in the cropped image
            pred = predict_digit(roi)

            # Print the prediction
            print(pred)

            # Place a label on the image with the predicted digit
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            img_org = put_label(img_org, pred, x, y)

    return img_org
