import cv2
import numpy as np
from detect_path import perform_detection
from detect_path import detect_persons


SIMILARITY_SCORE_THRESHOLD = 10
NUM_FEATURES = 512
NOR_X = 300
NOR_Y = 300


# create a opencv sift extractor
def get_sift():
    return cv2.xfeatures2d.SIFT_create(nfeatures=NUM_FEATURES, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)

index_dict = {}
sift = get_sift()
bf = cv2.BFMatcher()


# calculate sift
def calc_sift(sift, image_o):

    image = cv2.resize(image_o, (NOR_X, NOR_Y))
    if image.ndim == 2:
        gray_image = image
    else:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kp, des = sift.detectAndCompute(gray_image, None)

    #sift_feature = np.matrix(des)
    return kp, des


def get_sift_features(image):
    kp, sift_feature = calc_sift(sift, image)
    return sift_feature

def compare_sift(desc1, desc2):
    matches = bf.knnMatch(desc1,desc2, k=2)
    #matches = bf.match(des_current_siftfeature,sift_feature)
    #matches = sorted(matches, key=lambda val: val.distance)
     # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    distance = 0
    distance = (len(good) / (min(len(desc1), len(desc2))))
    #distance = 1 - (len(good) / len(sift_feature))
    #for j in range(len(good)):
    #    distance += good[j][0].distance 
    return distance * 100

def isRectangleOverlap(Person_rect, Roi_rect):
    rx, ry, rx2, ry2 = Person_rect
    px, py, px2, py2 = Roi_rect
    #if px >= rx2 or py >= ry2 or px2 <= rx or py2 <= ry:
    #if (px >= rx or px <= rx2) or (py >= ry or py <= ry2) or (px2 >= rx or px2 <= rx2) or (py2 >= ry or py2 <= ry2):
    if (px >= rx and px <= rx2 and py >= ry and py <= ry2) or (py >= ry and py <= ry2 and px2 >= rx and px2 <= rx2) or (px2 >= rx and px2 <= rx2 and py2 >= ry and py2 <= ry2) or (py2 >= ry and py2 <= ry2 and px >= rx and px <= rx2):
        return True
    return False
    
def is_overlapping_1D(line1, line2):
    """
    line:
        (xmin, xmax)
    """
    return line1[0] <= line2[1] and line2[0] <= line1[1]

def is_overlapping_2d(box1, box2):
    """
    box:
        (xmin, ymin, xmax, ymax)
    """
    return is_overlapping_1D([box1[0],box1[2]],[box2[0],box2[2]]) and is_overlapping_1D([box1[1],box1[3]],[box2[1],box2[3]])

from shapely.geometry import Polygon
def overlap2(box1, box2):
    p1 = Polygon([(box1[0],box1[1]), (box1[0],box1[3]), (box1[2],box1[3]),(box1[2],box1[1])])
    p2 = Polygon([(box2[0],box2[1]), (box2[0],box2[3]), (box2[2],box2[3]),(box2[2],box2[1])])
    return p1.intersects(p2)

def intersects(box1, box2):
    return not (box1[2] < box2[0] or box1[0] > box2[2] or box1[1] > box2[3] or box1[3] < box2[1])
    
# cases? 
# 1- if products not detected in reference image
# 2- if products not detected in current image
# 3- comparison accuracy
# API method
def check_panogram(current_image, reference_image, roi_list,prodcut_inside_size):
    
    persons = detect_persons(current_image)
    #if persons is not None and len(persons) > 0:
    #    return []
    #step2: perform detection on reference image
    reference_products = perform_detection(reference_image)
    
    #step3: perform detection on current image
    current_products = perform_detection(current_image)

    # checks
    if len(reference_products) == 0: 
        # TODO: what to do if products not detected in reference image?
        return []

    if len(current_products) == 0:
        # TODO: assume that shelf is empty, NO PLANOGRAM
        # TODO: future checks using image processing / ML
        return []

    print('Detections: ' + str(len(current_products)))
    roi_results={}
    if len(roi_list) > 0:
        kz=-1
        for roi_id,roi_rectangle in roi_list.items(): # step1: loop over all ROIs
            kz=kz+1
            result = []
            person_found = False
            #cv2.rectangle(current_image, (roi_rectangle[0],roi_rectangle[1]), (roi_rectangle[0] + roi_rectangle[2], roi_rectangle[1] + roi_rectangle[3]), (0,0,255), 2)
            if persons is not None and len(persons) > 0:
                
                for p in range(len(persons)):
                    px, py, pw, ph = persons[p]
                    #cv2.rectangle(current_image, (px, py), (px+pw, py+ ph), (255,255,0), 2)
                    if intersects([px, py, px+pw, py+ph], [roi_rectangle[0], roi_rectangle[1], roi_rectangle[0] + roi_rectangle[2], roi_rectangle[1] + roi_rectangle[3]]):
                        person_found = True
                        break
            if person_found:
                #print('skip ' + roi_id)
                continue
            for j in range(len(current_products)):
                x, y, w, h = current_products[j]
                # check if product is inside the current ROI in the current frame
                if prodcut_inside_size[kz]<=0:
                    prodcut_inside_size[kz]=0.05
                elif prodcut_inside_size[kz]>1:
                    prodcut_inside_size[kz]=1
                    
                if x >= roi_rectangle[0] and (x+w*prodcut_inside_size[kz]) <= roi_rectangle[0] + roi_rectangle[2] and y >= roi_rectangle[1] and (y+h*prodcut_inside_size[kz]) <= roi_rectangle[1] + roi_rectangle[3]:
                    cv2.rectangle(current_image, (x,y), (x+w, y+h), (0,255,0), 2)
                    current_product_image = current_image[y:y+h, x:x+w]
                    current_product_features = get_sift_features(current_product_image)
                    max_score = 0
                    for k in range(len(reference_products)):
                        x1, y1, w1, h1 = reference_products[k]

                        # check if product is inside the current ROI in the reference frame 
                        # #step4: comparison
                        if x1 >= roi_rectangle[0] and x1 <= roi_rectangle[0] + roi_rectangle[2] and y1 >= roi_rectangle[1] and y1 <= roi_rectangle[1] + roi_rectangle[3]:
                            reference_product_image = reference_image[y1:y1+h1, x1:x1+w1]

                            # calcualte similarity
                            ref_product_features = get_sift_features(reference_product_image)
                            similarity_score = compare_sift(current_product_features, ref_product_features)
                            if similarity_score > max_score:
                                max_score = similarity_score
                            if similarity_score >= SIMILARITY_SCORE_THRESHOLD:
                                # NO PLANOGRAM
                                break
                    
                    # check if current product 
                    if max_score < SIMILARITY_SCORE_THRESHOLD:
                        # PLANOGRAM
                        # TODO: APPEND ROI ID
                        result.append([x, y, w, h])
            roi_results[roi_id] = result

            
    return roi_results       


