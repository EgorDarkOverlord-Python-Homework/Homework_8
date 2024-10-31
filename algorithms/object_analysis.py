"""
Шаблонный метод (Template method)
"""
import random
import cv2
import numpy as np

class ObjectAnalysis(object):
    def template_method(self, image):

        image = self.noise_filtering(image)
        data = self.segmentation(image)
        data = self.object_parameters(data)

        return data

    def noise_filtering(self, image):
        raise NotImplementedError()

    def segmentation(self, data):
        raise NotImplementedError()

    def object_parameters(self, data):
        (image, data) = data
        (numLabels, labels, stats, centroids) = data
        x = []
        y = []
        w = []
        h = []
        area = []
        for i in range(1, numLabels):
            # extract the connected component statistics for the current
            # label
            x.append(stats[i, cv2.CC_STAT_LEFT])
            y.append(stats[i, cv2.CC_STAT_TOP])
            w.append(stats[i, cv2.CC_STAT_WIDTH])
            h.append(stats[i, cv2.CC_STAT_HEIGHT])
            area.append(stats[i, cv2.CC_STAT_AREA])

        return (x, y, w, h, area)


class BinaryImage(ObjectAnalysis):
    def __init__(self):
        pass

    def noise_filtering(self, image):
        median = cv2.medianBlur(image, 5)
        return median

    def segmentation(self, image):
        output = cv2.connectedComponentsWithStats(
            image,
            4, # connectivity
            cv2.CV_32S)
        return (image, output)

class MonochromeImage(BinaryImage):
    def __init__(self):
        pass

    def noise_filtering(self, image):
        gaussian = cv2.GaussianBlur(image, (5, 5), 0)
        return gaussian

    def segmentation(self, image):
        output = cv2.Canny(image,100,200)
        return (image, output)

    def object_parameters(self, data):
        return super().object_parameters(data)
    
class ColorImage(MonochromeImage):
    def __init__(self):
        pass

    def noise_filtering(self, image):
        return cv2.bilateralFilter(image,9,75,75)
    
    def segmentation(self, image):      
        #Black background image
        src = image
        src[np.all(src == 255, axis=2)] = 0

        kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)

        # do the laplacian filtering as it is
        # well, we need to convert everything in something more deeper then CV_8U
        # because the kernel has some negative values,
        # and we can expect in general to have a Laplacian image with negative values
        # BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
        # so the possible negative number will be truncated
        imgLaplacian = cv2.filter2D(src, cv2.CV_32F, kernel)
        sharp = np.float32(src)
        imgResult = sharp - imgLaplacian

        # convert back to 8bits gray scale
        imgResult = np.clip(imgResult, 0, 255)
        imgResult = imgResult.astype('uint8')
        imgLaplacian = np.clip(imgLaplacian, 0, 255)
        imgLaplacian = np.uint8(imgLaplacian)

        bw = cv2.cvtColor(imgResult, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(bw, 40, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        cv2.imshow('Binary Image', bw)
             
        dist = cv2.distanceTransform(bw, cv2.DIST_L2, 3)
        
        # Normalize the distance image for range = {0.0, 1.0}
        # so we can visualize and threshold it
        cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)

        _, dist = cv2.threshold(dist, 0.4, 1.0, cv2.THRESH_BINARY)
 
        # Dilate a bit the dist image
        kernel1 = np.ones((3,3), dtype=np.uint8)
        dist = cv2.dilate(dist, kernel1)

        dist_8u = dist.astype('uint8')
 
        # Find total markers
        contours, _ = cv2.findContours(dist_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create the marker image for the watershed algorithm
        markers = np.zeros(dist.shape, dtype=np.int32)
        
        # Draw the foreground markers
        for i in range(len(contours)):
            cv2.drawContours(markers, contours, i, (i+1), -1)
        
        # Draw the background marker
        cv2.circle(markers, (5,5), 3, (255,255,255), -1)
        markers_8u = (markers * 10).astype('uint8')   
        
        cv2.watershed(imgResult, markers)
        
        #mark = np.zeros(markers.shape, dtype=np.uint8)
        mark = markers.astype('uint8')
        mark = cv2.bitwise_not(mark)
        
        # Generate random colors
        colors = []
        for contour in contours:
            colors.append((random.randint(0,256), random.randint(0,256), random.randint(0,256)))
        
        # Create the result image
        dst = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)
        
        # Fill labeled objects with random colors
        for i in range(markers.shape[0]):
            for j in range(markers.shape[1]):
                index = markers[i,j]
                if index > 0 and index <= len(contours):
                    dst[i,j,:] = colors[index-1]
        
        return (image, dst)
        
    def object_parameters(self, data):
        return super().object_parameters(data)

"""
Декоратор - структурный паттерн
"""

class FilteredAnalysis(ObjectAnalysis):
    def __init__(self, obj):
        self._proc = obj

    def template_method(self, image):
        (_x, _y, _w, _h, _area) = self._proc.template_method(image)
        x = []
        y = []
        w = []
        h = []
        area = []

        for i in range(len(_area)):
            if _area[i] > 10 and _area[i] < 2500:
                x.append(_x[i])
                y.append(_y[i])
                w.append(_w[i])
                h.append(_h[i])
                area.append(_area[i])

        return (x,y,w,h,area)


if __name__== '__main__':
    print("Binary Image Processing")
    bin_segm = BinaryImage()
    (x,y,w,h,area) = bin_segm.template_method(cv2.imread('./data/1.jpg', cv2.IMREAD_GRAYSCALE))
    for i in range(len(area)):
            print([x[i], y[i], w[i],h[i],area[i]])

    print("Decorated Binary Image Processing")
    filt_bin = FilteredAnalysis(BinaryImage())
    (x, y, w, h, area) = filt_bin.template_method(cv2.imread('./data/1.jpg', cv2.IMREAD_GRAYSCALE))
    for i in range(len(area)):
            print([x[i], y[i], w[i],h[i],area[i]])
