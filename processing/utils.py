import numpy as np
import cv2
import imutils
import glob
import os

idx = 0

def processing_alternative(img,img_copy):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    edged = cv2.Canny(blur, 10, 200)

    cont, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    extracted_characters = []

    idx = 0
    for c in sort_contours(cont):
        (x, y, w, h) = cv2.boundingRect(c)
        ratio = h / w
        inv_ratio = w/h
        ret1, test = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)
        mul = w*h
        roi = test[y:y + h, x:x + w]

        if 1.3 <= ratio <= 8 and inv_ratio > 0.3 and mul <10000:
            if h / img_copy.shape[0] >= 0.091:
                if x>50 and y>90 and y<350:
                    idx += 1
                    cv2.rectangle(img_copy, (x, y), (x + w, y + h), (100, 0, 255), 2)
                    extracted_characters.append(roi)

        path = "detect/"

        if len(extracted_characters) > 7:
            del extracted_characters[0]

        for i in range(len(extracted_characters)):
            cv2.imwrite(os.path.join(path, str(i) + '.jpg'), extracted_characters[i])

    print("Detect {} letters:".format(len(extracted_characters)))
    if len(extracted_characters) == 0:
        return('???????')
    else:
        word = detect()
        return word

def detect():
    chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','A', 'B', 'C','D', 'E', 'F', 'G', 'H', 'I', 'J',
                     'K', 'L', 'M', 'N', 'O', 'P','R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    chars_dic = {}
    i = 0
    images_detect = []
    dsize = (40,60)

    # Loading cut characters
    for img_detect in glob.glob("detect/*.jpg"):
         n = cv2.imread(img_detect, cv2.IMREAD_GRAYSCALE)
         images_detect.append(n)

    result = []
    result_end = []
    #assigning characters to photos
    for image_path in glob.glob("wzorce/*.png"):
        chars_dic[image_path] = chars[i]
        i+=1

    for l, img in enumerate(images_detect):
        img = cv2.resize(img, dsize)
        slownik = {}
        for image_path in glob.glob("wzorce/*.png"):
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, dsize)
            if image is None:
                print(f'Error loading image {image_path}')
                continue
            score = 0
            max_score = 0
            for i in range(40):
                for j in range(60):
                    if img[j][i] == image[j][i]:
                        score = score + 1
            if score > max_score:
                max_score = score
                best_ident = image_path
                slownik[max_score] = best_ident

            #exceptions:
            if (l > 2) and (chars_dic[slownik[max(slownik)]] == 'B'):
                chars_dic[slownik[max(slownik)]]= '8'
            if (l > 2) and (chars_dic[slownik[max(slownik)]] == 'D'):
                chars_dic[slownik[max(slownik)]] = '0'
            if (l > 2) and (chars_dic[slownik[max(slownik)]] == 'I'):
                chars_dic[slownik[max(slownik)]] = '1'
            if (l > 2) and (chars_dic[slownik[max(slownik)]] == 'O'):
                chars_dic[slownik[max(slownik)]] = '0'
            if (l > 2) and (chars_dic[slownik[max(slownik)]] == 'Z'):
                chars_dic[slownik[max(slownik)]] = '2'

        result.append(chars_dic[slownik[max(slownik)]])

    if len(result)>2:
        if result[1] == 'O' and result[2] == 'O':
            result[2] = '0'

    if len(result)<7:
        for i in range(7-len(result)):
            result_end.append('?')
    for i in range(len(result)):
        result_end.append(result[i])
    word = "".join(result_end)

    for img_detect in glob.glob("detect/*.jpg"):
        os.remove(img_detect)

    return word

def sort_contours(contours, reverse=False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    (cnts, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts

def sort(data):
    for i in range(len(data) - 1, 0, -1):
        for j in range(i):
            if data[j] > data[j + 1]:
                data[j], data[j + 1] = data[j + 1], data[j]

def perform_processing(image: np.ndarray) -> str:
    img = image.copy()
    #print(f'image.shape: {img.shape}')
    img = cv2.resize(img, (640, 480))
    img_copy = img.copy()
    # convert to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur to reduce noise
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    # Edge detection
    edged = cv2.Canny(blur, 10, 200)

    # find contours
    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:4]
    points = None
    # loop over contours
    for c in contours:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*perimeter, True)

        if len(approx) == 4:
            points = approx
            break

    if points is None:
        flag = False
        #print("No contour detected")
    else:
        flag = True

    if flag == False:
        #cv2.imshow('IMAGE', img)
        result_alt = processing_alternative(img,img_copy)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        print(result_alt)
        return result_alt
    if flag == True:
        cv2.drawContours(img, [points], -1, (0, 255, 0), 3)
        #cv2.imshow('IMAGE', img)

    # Masking the part other than the number plate
    mask = np.zeros(gray.shape, np.uint8)
    mask_image = cv2.drawContours(mask, [points], 0, 255, -1, )
    mask_image = cv2.bitwise_and(img, img, mask=mask)
    #cv2.imshow('mask',mask_image)

    (x, y) = np.where(mask == 255)
    (min_x, min_y) = (np.min(x), np.min(y))
    (max_x, max_y) = (np.max(x), np.max(y))
    plate_image = img_copy[min_x:max_x + 1, min_y:max_y + 1]

    whidth = plate_image.shape[1]
    height = plate_image.shape[0]

    #image straightening
    dst_points = np.float32([[0,0],[0,height],[whidth,height],[whidth,0]])
    src_points = np.ones_like(dst_points)
    src_sorted = np.ones_like(dst_points)

    list_x = []
    list_y = []

    for i in range(4):
        src_points[i] = points[i]
        list_x.append(src_points[i][0])
        list_y.append(src_points[i][1])

        sort(list_x)
        sort(list_y)

    for i in range(4):
        if (src_points[i][0] == list_x[0] and src_points[i][1] == list_y[0]) \
                or (src_points[i][0] == list_x[len(list_x)-3] and src_points[i][1] == list_y[0]) \
                or (src_points[i][0] == list_x[0] and src_points[i][1] == list_y[len(list_y)-3]) \
                or (src_points[i][0] == list_x[len(list_x)-3] and src_points[i][1] == list_y[len(list_y)-3]):
            src_sorted[0] = src_points[i]
        elif (src_points[i][0] == list_x[0] and src_points[i][1] == list_y[len(list_y)-1]) \
                or (src_points[i][0] == list_x[len(list_x)-3] and src_points[i][1] == list_y[len(list_y)-1]) \
                or (src_points[i][0] == list_x[0] and src_points[i][1] == list_y[len(list_y)-2]) \
                or (src_points[i][0] == list_x[len(list_x)-3] and src_points[i][1] == list_y[len(list_y)-2]):
            src_sorted[1] = src_points[i]
        elif (src_points[i][0] == list_x[len(list_x)-1] and src_points[i][1] == list_y[len(list_y)-1]) \
                or (src_points[i][0] == list_x[len(list_x)-2] and src_points[i][1] == list_y[len(list_y)-1]) \
                or (src_points[i][0] == list_x[len(list_x)-1] and src_points[i][1] == list_y[len(list_y)-2]) \
                or (src_points[i][0] == list_x[len(list_x)-2] and src_points[i][1] == list_y[len(list_y)-2]):
            src_sorted[2] = src_points[i]
        elif (src_points[i][0] == list_x[len(list_x)-1] and src_points[i][1] == list_y[0]) \
                or (src_points[i][0] == list_x[len(list_x)-2] and src_points[i][1] == list_y[0]) \
                or (src_points[i][0] == list_x[len(list_x)-1] and src_points[i][1] == list_y[len(list_y)-3]) \
                or (src_points[i][0] == list_x[len(list_x)-2] and src_points[i][1] == list_y[len(list_y)-3]):
            src_sorted[3] = src_points[i]

    iter = 0

    for i in range(len(src_sorted)):
         if src_sorted[i][0] != 1. and src_sorted[i][1] != 1.:
             iter+=1

    if iter == 4:
        perspective_transform = cv2.getPerspectiveTransform(src_sorted, dst_points)
        perspective_plate = cv2.warpPerspective(img_copy, perspective_transform, (whidth, height))
        #cv2.imshow('Plate_perspective_transform', perspective_plate)
    else:
        perspective_plate = plate_image

    # number plate processing
    gray_p = cv2.cvtColor(perspective_plate, cv2.COLOR_BGR2GRAY)
    blur_p = cv2.bilateralFilter(gray_p, 11, 17, 17)
    ret, binary = cv2.threshold(blur_p,100, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((5, 5), np.uint8)
    thre_mor = cv2.erode(binary, kernel, iterations=1)
    open_morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, (5,5))

    ret1, test = cv2.threshold(gray_p,100, 255, cv2.THRESH_BINARY)
    kernel1 = np.ones((3, 3), np.uint8)
    test = cv2.erode(test, kernel1, iterations=1)
    test = cv2.morphologyEx(test, cv2.MORPH_CLOSE, (3,3))


    cont, _ = cv2.findContours(open_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    plate_copy = perspective_plate.copy()

    extracted_characters = []

    idx = 0

    for c in sort_contours(cont):
        (x, y, w, h) = cv2.boundingRect(c)
        ratio = h / w
        roi = test[y:y + h, x:x + w]
        if 1 <= ratio <= 6:
            if h / perspective_plate.shape[0] >= 0.4:
                idx += 1
                cv2.rectangle(plate_copy, (x, y), (x + w, y + h), (100, 0, 255), 2)
                extracted_characters.append(roi)

        path = "detect/"

        if len(extracted_characters) > 7:
            del extracted_characters[0]

        for i in range(len(extracted_characters)):
            cv2.imwrite(os.path.join(path, str(i) + '.jpg'), extracted_characters[i])

    print("Detect {} letters:".format(len(extracted_characters)))
    if len(extracted_characters) == 0:
         return('???????')

    word = detect()
    #print(word)

    return word