import numpy as np
import cv2
from vector import *

from scipy import ndimage
from load import model_create
import tensorflow as tf
from vector import distance, pnt2line


#model = model_create()



#detektujemo liniju sve ok
def houghLinesTransformation(img):
   # img = cv2.imread('dave.jpg')

   # edges = cv2.Canny(gray, 50, 150, apertureSize=3)

   # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   # erosion = cv2.erode(gray, kernel_line, iterations=1)
   # edges = cv2.Canny(erosion, 50, 150, 3)
    #blur = cv2.GaussianBlur(img, (5, 5), 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel_line = np.ones((2, 2), np.uint8)
    erosion = cv2.erode(gray, kernel_line, iterations=1)
    #edges = cv2.Canny(erosion, 50, 150, 3)
    #edges = cv2.Canny(erosion, 50, 150, 3)
    image_bin = cv2.threshold(erosion,20, 255, cv2.THRESH_BINARY)[1]
    minLineLength = 100
    maxLineGap = 100

    lines= cv2.HoughLinesP(image_bin, 1, np.pi / 180, 100, minLineLength, maxLineGap)

    return lines


def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret,image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin


def find_liness_coord(liness):
    if liness is not None:
        for x1, y1, x2, y2 in liness[0]:
            line = x1, y1, x2, y2
            return line
    else:
        return None


def resize_region(region):
    #'''Transformisati selektovani region na sliku dimenzija 28x28'''
    return cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)




def inRange(r, item):
    retVal = []
    for obj in global_contours:
        x1, y1, w1, h1= cv2.boundingRect(item)
        x2, y2, w2, h2 = cv2.boundingRect(obj)
        mdist = distance((x1, y1),(x2, y2))

        if(mdist<r):
            return True
    global_contours.append(item)
    return False

global_contours=[]

def select_roi(image_orig, image_bin):
    contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    regions_array = []
    elements = []  # lista sortiranih regiona po x osi (sa leva na desno)
    digits = []
    #lines = houghLinesTransformation(image_orig)


    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # koordinate i velicina granicnog pravougaonika
        area = cv2.contourArea(contour)

        cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if area > 6 and h < 48 and h > 10 and w > 2:
            center = (x + w / 2, y + h / 2)
            # kopirati [y:y+h+1, x:x+w+1] sa binarne slike i smestiti u novu sliku
            # označiti region pravougaonikom na originalnoj slici (image_orig) sa rectangle funkcijom
            #image_bin = cv2.medianBlur(image_bin, 5)
            region = image_bin[y-16:y + 30, x-16:x + 30]

            lines = houghLinesTransformation(image_orig)
            if lines is not None:
                for x1, y1, x2, y2 in lines[0]:
                    #cv2.line(image_orig, (x1, y1), (x2, y2), (192, 0, 182), 2)
                    #cv2.line(image_orig, (x1, y1), (x2, y2), (192, 0, 182), 2)

                    dist, pt, r = pnt2line2((x + w, y + h), (x1, y1), (x2, y2))
                    #print(dist)
                    #cv2.imshow('imgreg', region)
                    #cv2.waitKey(0)
                    if dist < 6 and r==1:
                        if(inRange(18, contour)==False):

                            #print(dist)
                            #print(regions_array)
                            cv2.imshow('imgreg', region)
                            cv2.waitKey(0)

                            regions_array.append([resize_region(region), (x, y, w, h)])


               # detected_digit = detect_digits(digits, digit)

    # sortirati sve regione po x osi (sa leva na desno) i smestiti u promenljivu sorted_regions


    return image_orig, regions_array


def scale_to_range(image): # skalira elemente slike na opseg od 0 do 1
    #''' Elementi matrice image su vrednosti 0 ili 255.
    #    Potrebno je skalirati sve elemente matrica na opseg od 0 do 1
    #'''
    return image/255

def matrix_to_vector(image):
    #'''Sliku koja je zapravo matrica 28x28 transformisati u vektor sa 784 elementa'''
    return image.flatten()


numbers = range(0, 40)
evens = numbers[2::2]


img = cv2.imread('houghlines5.jpg')
lines=houghLinesTransformation(img)
for x1, y1, x2, y2 in lines[0]:
    cv2.line(img, (x1, y1), (x2, y2), (192, 0, 182), 2)
cv2.imwrite('houghlines66.jpg', img)



#ucitavanje modela
model = tf.keras.models.load_model('model/model2.h5')



for i in range(10):
    # if not i == 0:
    #     continue
    video_name = 'video-' + str(i) + '.avi'
    path_to_video = 'videos/' + video_name
    cap = cv2.VideoCapture(path_to_video)
    cap2 = cv2.VideoCapture("videos/video-2.avi")
    ret, frame = cap.read()
    frame_num = 0
    b, g, r = cv2.split(frame)
    print(video_name)

    konacno = []
    numbers = []

    sum_digits = 0

    while (True):
        ret, frame = cap.read()
        frame_num += 1



        if ret is not True:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        imgb = image_bin(gray)
        cv2.imwrite('houghlines6637.jpg', imgb)
        #digits_frame = get_digits_frame(frame)
        #contours = cv2.findContours(digits_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #za frame detektujem liniju

        #sum_digits=get_sum_of_digits(frame)
        picture, region = select_roi(frame, imgb)

        ready_for_ann = []
        #display_image(picture)
        for r in region:
            broj = r[0].reshape(28, 28)
            #cv2.imshow('frame2r', broj)
            #cv2.waitKey(0)
            priprema=matrix_to_vector(scale_to_range(broj))
            pred=model.predict(priprema.reshape(1,28,28,1))
            print(pred.argmax())
            sum_digits+=pred.argmax()

        cv2.imshow('frame2', frame)

        if cv2.waitKey(1) == 30:
            break

    print(frame_num)
    print(video_name)
    print(sum_digits)
    cap.release()
    cv2.destroyAllWindows()
    #  sum = get_sum_of_digits(path_to_video)
    #  prediction_results.append({ 'video': video_name, 'sum': sum })


print(evens)
cv2.imwrite('houghlines66.jpg', img)