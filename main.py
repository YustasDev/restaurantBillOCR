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

def perspective_transformation(input_file):

    img = cv2.imread(input_file)
    rows, cols, ch = img.shape
    pts1 = np.float32([[60, 10], [787, 21], [54, 1104], [841, 1077]])
    pts2 = np.float32([[0, 0], [841, 0], [0, 1104], [841, 1104]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (850, 1105))

    correctedFile = 'correctedImg.jpg'
    cv2.imwrite(correctedFile, dst)
    #cv2.imshow('corrected', dst)
    return correctedFile

def lighting_correction(input_file):
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    # img = clahe.apply(gray)
    # cv2.imwrite(f"{target_path}.test.jpg", img)


    img = cv2.imread(input_file)
    # gray scale image
    RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    R, G, B = cv2.split(RGB)

    # Create a CLAHE object: The image is divided into small block 8x8 which they are equalized as usual.
    clahe = cv2.createCLAHE(
        clipLimit=0.1, tileGridSize=(8, 8))  # clipLimit=2.5

    # Applying this method to each channel of the color image
    output_2R = clahe.apply(R)
    output_2G = clahe.apply(G)
    output_2B = clahe.apply(B)

    # mergin each channel back to one
    img_output = cv2.merge((output_2R, output_2G, output_2B))

    # coverting image from RGB to Grayscale
    # eq = cv2.cvtColor(img_output, cv2.COLOR_BGR2GRAY)

    # Using image thresholding to classify pixels as dark or light
    # This method provides changes in illumination and the contrast of the image is improved.
    # gauss = cv2.adaptiveThreshold(
    #     eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 45)  # 11, 2

    output_path = 'Img_WithLighting_correction.jpg'
    cv2.imwrite(output_path, img_output)

    return output_path



if __name__ == '__main__':

    input_file = './Images/bill_02.jpg'
    correctedFile = perspective_transformation(input_file)
    image_withLighting_correction = lighting_correction(correctedFile)



    img = cv2.imread(image_withLighting_correction, cv2.IMREAD_GRAYSCALE)
    ret, img_binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite('bill_02_gray.jpg', img_binary)
    imageGray = Image.open('bill_02_gray.jpg')


    threshold = 0.915
    #img = Image.open(image_withLighting_correction)
    result_rows = find_lightest_rows(imageGray, threshold)
    sorted_rows = sorted(result_rows)
    print(sorted_rows)

    separatingLines = []
    referenceNum = 0
    separatingLines.append(referenceNum)
    for separatingLine in sorted_rows:
        if(separatingLine > (referenceNum + 5)):
            separatingLines.append(separatingLine)
        referenceNum = separatingLine

    print('separatingLines: ')
    print(separatingLines)


    max_column = 841
    resultImage = cv2.imread('bill_02_gray.jpg')
    #imgWithLines = cv2.imread(image_withLighting_correction)
    #resultImage = cv2.cvtColor(imgWithLines, cv2.COLOR_BGR2RGB)
    #for row in sorted_rows:
    for row in separatingLines:
        start_point = (0, row)
        end_point = (max_column, row)
        cv2.line(resultImage, start_point, end_point, (0, 0, 255), thickness=1)

    cv2.imshow('Result', resultImage)
    cv2.waitKey()






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

    """
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
    """

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
