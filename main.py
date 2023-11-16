import os
import tempfile
import time
import csv
import cv2
from PIL import Image
import easyocr
import numpy as np
from scipy.ndimage import interpolation as inter


def normalize(img):
    norm_img = np.zeros((img.shape[0], img.shape[1]))
    img = cv2.normalize(img, norm_img, 0, 255, cv2.NORM_MINMAX)
    return img

def correct_skew(image, delta=1, limit=45):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return histogram, score

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    corrected = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
            borderMode=cv2.BORDER_REPLICATE)

    return best_angle, corrected

def find_lightest_rows(img, threshold):
    line_luminances = [0] * img.height

    for y in range(img.height):
        for x in range(img.width):
            #line_luminances[y] += img.getpixel((x, y))[0]     # for BGR/RGB etc.
            line_luminances[y] += img.getpixel((x, y))

    line_luminances = [x for x in enumerate(line_luminances)]
    line_luminances.sort(key=lambda x: -x[1])
    lightest_row_luminance = line_luminances[0][1]
    lightest_rows = []
    for row, lum in line_luminances:
        if(lum > lightest_row_luminance * threshold):
            lightest_rows.append(row)

    return lightest_rows


def find_score(arr, angle):
    data = inter.rotate(arr, angle, reshape=False, order=0)
    hist = np.sum(data, axis=1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return hist, score

def correct_rotate(input_file):
    img = Image.open(input_file)

    # convert to binary
    wd, ht = img.size
    pix = np.array(img.convert('1').getdata(), np.uint8)
    bin_img = 1 - (pix.reshape((ht, wd)) / 255.0)
    delta = 0.5
    limit = 2
    angles = np.arange(-limit, limit + delta, delta)
    scores = []
    for angle in angles:
        hist, score = find_score(bin_img, angle)
        scores.append(score)

    best_score = max(scores)
    best_angle = angles[scores.index(best_score)]
    print('Best angle: ', best_angle)
    # correct skew
    data = inter.rotate(bin_img, best_angle, reshape=False, order=0)
    img = Image.fromarray((255 * data).astype("uint8")).convert("RGB")
    img.save('skew_corrected.png')

    image = cv2.imread('skew_corrected.png')
    img = cv2.bitwise_not(image)
    cv2.imshow("Image", img)
    cv2.imwrite('bill_02_rotate.jpg', img)



if __name__ == '__main__':

    # image = cv2.imread('bill_02.jpg')
    # #img_large = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    #threshold = 0.99
    #
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # ret, threshGray = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    #
    # cv2.imwrite('bill_02_gray.jpg', threshGray)
    #
    #imageGray = Image.open('bill_02_gray.jpg')
    # cv2.waitKey()
    # result_rows = find_lightest_rows(imageGray, threshold)
    # print(result_rows)



    #================= angle detection test ===============>

    #correct_rotate('bill_02.jpg')

    # image = cv2.imread('test_angle2.png')
    # angle, corrected = correct_skew(image)
    # print('Skew angle:', angle)
    # cv2.imshow('corrected', corrected)
    # cv2.waitKey()


    reader = easyocr.Reader(['ru', 'en'], gpu=False)
    #result = reader.readtext('bill_02.jpg', detail=0)

    list_input = [r'./Images/bill_01_str.jpg', r'./Images/bill_011_str.jpg', r'./Images/bill_012_str.jpg']
    for index, element in enumerate(list_input):
        image = cv2.imread(element)
        image = normalize(image)

    #=====================================================>

    # angle, corrected = correct_skew(image)
    # print('Skew angle:', angle)



    #====================================================<

        result = reader.readtext(image)

        # loop over the results
        for (bbox, text, prob) in result:
            # display the OCR'd text and associated probability
            print("[INFO] {:.4f}: {}".format(prob, text))
            # unpack the bounding box
            (tl, tr, br, bl) = bbox
            tl = (int(tl[0]), int(tl[1]))
            tr = (int(tr[0]), int(tr[1]))
            br = (int(br[0]), int(br[1]))
            bl = (int(bl[0]), int(bl[1]))
            # draw the box surrounding the text along
            # with the OCR'd text itself)
            cv2.rectangle(image, tl, br, (0, 255, 0), 2)
            cv2.putText(image, text, (tl[0], tl[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        # show the output image
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        print(result)


    # print(pytesseract.image_to_string(Image.open('bill_02.jpg'), lang='rus'))
    #
    # img_cv = cv2.imread(r'bill_02.jpg')
    # img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    # print(pytesseract.image_to_string(img_rgb, lang='rus'))


    #
    #
    # output_dir = init_output_folder('output')
    #
    # images = [('bill_01.jpg', 'rus')]
    #
    # for bill in images:
    #     img_path, lang = bill
    #
    #     print(f"Process: {img_path}")
    #     target_path = os.path.join(output_dir, os.path.basename(img_path))
    #
    #     img = cv2.imread(img_path)
    #
    #     get_boxes(img, target_path)
    #     get_data(img, target_path)
    #
    #
