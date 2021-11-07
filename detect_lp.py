import cv2
import imutils
import numpy as np


# Ham sap xep contour tu trai sang phai

def get_contour_precedence(contour, cols):
    tolerance_factor = 10
    origin = cv2.boundingRect(contour)
    return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]


def sort_contours(cnts):
    cnts.sort(key=lambda x:get_contour_precedence(x, img.shape[1]))
    return cnts

# Ham fine tune bien so, loai bo cac ki tu khong hop ly
def fine_tune(lp):
    newString = ""
    for i in range(len(lp)):
        if lp[i] in char_list:
            newString += lp[i]
    return newString

# Dinh nghia cac ky tu tren bien so
char_list =  '0123456789ABCDEFGHKLMNPRSTUVXYZ'

# Param
max_size = 8000
min_size = 200

# Load image
img = cv2.imread('plate/0006_06797_b.jpg', cv2.IMREAD_COLOR)

# Cau hinh tham so cho model SVM
digit_w = 30 # Kich thuoc ki tu
digit_h = 60 # Kich thuoc ki tu

model_svm = cv2.ml.SVM_load('svm.xml')

# Resize image
img = cv2.resize(img, (470 , 300))

# Edge detection
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grey scale
# gray = cv2.GaussianBlur(gray,(5,5),0)
# gray = cv2.bilateralFilter(gray, 11, 17, 17)  # Blur to reduce noise
edged = cv2.Canny(gray, 30, 200)  # Perform Edge detection
# find contours in the edged image, keep only the largest
# ones, and initialize our screen contour
cv2.imshow("Gray image",edged )
cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
screenCnt = None

# loop over our contours
for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)

    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.05 * peri, True)

    # if our approximated contour has four points, then
    # we can assume that we have found our screen

    if len(approx) == 4 and max_size > cv2.contourArea(c) > min_size:
        screenCnt = approx
        break
if screenCnt is None:
    detected = 0
    print ("No plate detected")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    detected = 1

if detected == 1:
    cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)

    # Masking the part other than the number plate
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1, )
    new_image = cv2.bitwise_and(img, img, mask=mask)

    # Now crop
    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x)+1, np.min(y)+1)
    (bottomx, bottomy) = (np.max(x)-1, np.max(y)-1)
    Cropped = gray[topx:bottomx + 1, topy:bottomy + 1]

    # Display image
    cv2.imshow('Input image', img)
    cv2.imshow('License plate', Cropped)


    copyCropped = Cropped.copy()
    # gray = cv2.GaussianBlur(copyCropped,(5,5),0)
    # gray = cv2.bilateralFilter(gray, 11, 17, 17)  # Blur to reduce noise
    copyCropped = cv2.GaussianBlur(copyCropped,(5,5),1)
    edged = cv2.Canny(copyCropped, 30, 200,apertureSize=3) 
    lines = cv2.HoughLines(edged,1,np.pi/180,200)

    # for line in lines:
    #     for rho,theta in line:
    #         a = np.cos(theta)
    #         b = np.sin(theta)
    #         x0 = a*rho
    #         y0 = b*rho
    #         x1 = int(x0 + 1000*(-b))
    #         y1 = int(y0 + 1000*(a))
    #         x2 = int(x0 - 1000*(-b))
    #         y2 = int(y0 - 1000*(a))

    #         cv2.line(copyCropped,(x1,y1),(x2,y2),(0,0,255),2)
    print(lines)
    cv2.imshow('Line', edged)


    roi = Cropped

    # Ap dung threshold de phan tach so va nen
    binary = cv2.threshold(Cropped, 190, 255,
                         cv2.THRESH_BINARY_INV)[1]

    cv2.imshow("Anh bien so sau threshold", binary)
    # Xoay ảnh
    # cv2.houghLines(binary)
    
    # Segment kí tự
    # kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
    # binary = cv2.dilate(binary,kernel3,iterations = 1)
    cont, _  = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    # cv2.imshow("Anh bien so sau bien doi ", thre_mor) 
    plate_info = ""
    num = 0 
    for c in sort_contours(cont):
        (x, y, w, h) = cv2.boundingRect(c)
        ratio = h/w
        if 1.5<=ratio<=5: # Chon cac contour dam bao ve ratio w/h
            if h/roi.shape[0]>=0.32 and h/roi.shape[0] <0.6 : # Chon cac contour cao tu 60% bien so tro len
                # Ve khung chu nhat quanh so
                
                cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Tach so va predict
                curr_num = binary[y:y+h,x:x+w]
                curr_num = cv2.copyMakeBorder(curr_num,1,1,1,1,cv2.BORDER_CONSTANT,value=[0,0,0])
                curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                num  = num +1
                cv2.imshow("Contour num " + str(num), curr_num)
                _, curr_num = cv2.threshold(curr_num, 30, 255, cv2.THRESH_BINARY)
                curr_num = np.array(curr_num,dtype=np.float32)
                curr_num = curr_num.reshape(-1, digit_w * digit_h)

                # Dua vao model SVM
                result = model_svm.predict(curr_num)[1]
                result = int(result[0, 0])

                if result<=9: # Neu la so thi hien thi luon
                    result = str(result)
                else: #Neu la chu thi chuyen bang ASCII
                    result = chr(result)

                plate_info +=result

    cv2.imshow("Cac contour tim duoc", roi)

    # Viet bien so len anh
    cv2.putText(img,fine_tune(plate_info),(50, 50), cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 0, 255), lineType=cv2.LINE_AA)
    # Hien thi anh
    print("Bien so=", plate_info)
    cv2.imshow("Hinh anh output",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


