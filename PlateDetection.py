import skimage.io as io

# Show the figures / plots inside the notebook
from skimage.color import rgb2gray,rgb2hsv
from skimage.exposure import equalize_adapthist 
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.draw import rectangle
import imutils
from skimage.exposure import histogram
from matplotlib.pyplot import bar

def show_images(images,titles=None):
    #This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n)
        if image.ndim == 2: 
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show() 
    
def read_image(image_name):
    img_data =cv2.imread(image_name)
    img_data = imutils.resize(img_data , width = 500)
    show_images ([img_data],["Original image"])
    return img_data

def image_preprocessing(img):
    '''
    in this function we are going to applying different preprocessing 
    techniques in the input image 
    prams:
            img 
    '''
    # initialize the rectangular and square kernels to be applied to the image,
    # then initialize the list of license plate regions
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    squareKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # convert the image to grayscale, and apply the blackhat operation
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)

    # find regions in the image that are light
    light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKernel)
    light = cv2.threshold(light, 50, 255, cv2.THRESH_BINARY)[1]

    # compute the gradient representation of the blackhat image and s
    sobelX = cv2.Sobel(blackhat,ddepth = cv2.CV_32F,dx = 1, dy = 0, ksize = -1)

    sobelX = np.absolute(sobelX)

    # scale theresulting image into the range [0, 255]
    (minVal, maxVal) = (np.min(sobelX), np.max(sobelX))
    sobelX = (255 * ((sobelX - minVal) / (maxVal - minVal))).astype("uint8")

    # blur the gradient representation, apply a closing operating, and threshold the
    # image using Otsu's method
    sobelX = cv2.GaussianBlur(sobelX, (9, 9), 0)
    sobelX = cv2.morphologyEx(sobelX, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(sobelX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # perform a series of erosions and dilations on the image
    thresh = cv2.erode(thresh, None, iterations = 2)
    thresh = cv2.dilate(thresh, None, iterations = 3)

    # take the bitwise 'and' between the 'light' regions of the image, then perform
    # another series of erosions and dilations
    thresh = cv2.bitwise_and(thresh, thresh, mask = light)
    thresh = cv2.dilate(thresh, None, iterations =6)
    thresh = cv2.erode(thresh, None, iterations = 8)
    blur = cv2.GaussianBlur(thresh,(5,5),0)

    return blur , gray
   

def ratioCheck (w , h):
    #getting area and aspect ratio for 
    area = w* h
    aspect_ratio = w/h 
    return  (1.7< aspect_ratio < 4.5 and 1000<area < 4850)
    
def checkChannels(window , w, h):
    #getting average value for each channel 
    avg_red_ch = np.sum(window[:,:,0])/ (w*h)  
    avg_green_ch = np.sum(window[:,:,1])/ (w*h)   
    avg_blue_ch = np.sum(window[:,:,2])/ (w*h) 
    return  avg_red_ch >100 and avg_green_ch>100 and avg_blue_ch>80 

def plate_criteria(window , w ,h):
    return  ratioCheck(w,h) and checkChannels(window ,w , h)


def plateDetection(edge_img , img ):
    # getting contours form edge image
    cnts, _ = cv2.findContours(edge_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    plates = []
    #looping over the contours to get possible plates 
    for i, c in enumerate(cnts):
        x,y,w,h = cv2.boundingRect(c)
        window =img[y:y+h, x:x+w]  
        
        #check plate criteria for each contour 
        if ( plate_criteria(window , w ,h)  ):
                plate = img[y-7:y+h+7, x-1:x+w+1]
                if plate.shape[0] and plate.shape[1] :
                    plates.append(plate)
    #writing plate in scannedplates file                     
    for i, plate in enumerate(plates):
        cv2.imwrite(f'plates/Plate{i}.jpg', plate )

    return   plates

          