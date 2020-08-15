
import cv2 
import numpy as np 
from tqdm import tqdm
from glob import glob
import random
import os
import sys
import json
import math
import copy
    
import logging
FORMAT_STRING = "%(levelname)-8s:%(name)-8s.%(funcName)-8s>> %(message)s"
logging.basicConfig(format=FORMAT_STRING)
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

from pprint import pprint, pformat

COLOR_LINE         = (212, 188,   0)
COLOR_LINE_SEGMENT = (000,  44, 221)
COLOR_CIRCLE       = (34,   87, 255)

CIRCLE_RADIUS = 24
LINE_SEGMENT_THICKNESS = 12
LINE_THICKNESS = 3

def imshow(name, img, resize_factor = 0.4):
    return cv2.imshow(name,
                      cv2.resize(img,
                                 (0, 0),
                                 fx=resize_factor,
                                 fy=resize_factor))

def slope(x1, y1,  x2, y2):
    if x2 != x1:
        return( (y2 - y1) / (x2 - x1) )
    else:
        return 'NA'
    
def extend_line_to_boundary(img, line):
    (x1, y1), (x2, y2) = line

    m = slope(x1, y1, x2, y2)
    log.debug('slope: {}'.format(m))

    h, w = img.shape[:2]
    if m != 'NA':
        px, py = 0, -(x1-0) * m + y1
        qx, qy = w, -(x2-w) * m + y2
    else:
        px , py = x1, 0
        qx , qy = x1, h

    px, py, qx, qy = [ int(i) for i in [px, py, qx, qy] ]
    log.debug('px, py, qx, qy: {}, {}, {}, {}'.format(px, py, qx, qy))

    return px, py, qx, qy
            
def mkdir_if_exist_not(name):
    if not os.path.isdir(name):
        return os.makedirs(name)
    
def rotate(image, angleInDegrees):
    h, w = image.shape[:2]
    img_c = (w / 2, h / 2)

    rot = cv2.getRotationMatrix2D(img_c, angleInDegrees, 1)

    rad = math.radians(angleInDegrees)
    sin = math.sin(rad)
    cos = math.cos(rad)
    b_w = int((h * abs(sin)) + (w * abs(cos)))
    b_h = int((h * abs(cos)) + (w * abs(sin)))

    rot[0, 2] += ((b_w / 2) - img_c[0])
    rot[1, 2] += ((b_h / 2) - img_c[1])

    outImg = cv2.warpAffine(image, rot, (b_w, b_h), flags=cv2.INTER_LINEAR, borderValue=(255,255,255))
    return outImg


class Vetti:
    STATE_LINE_DRAWING = -1
    
    def __init__(self, args, name, img,
                 scale_factor = 0.5,
                 unit_rotation = 0.1):

        self.args = args
        self.name = name
        self.filepath = '{}/{}.vetti'.format(self.args.output_dir, self.name)
        
        self.img = img
        self.source = self.img.copy()
        self.source_backup = self.img.copy()
        
        #should saved to file
        self.finished = False
        self.unit_rotation = unit_rotation
        
        # this is very important otherwise callback point
        # and drawn points will not match,
        # I dare you change this I double dare you!!
        self.scale_factor = int(1/scale_factor) 

        self.rotation = 0

        self.line = [(100, 100), (1500, 2000)]

        self.first_point = None
        self.first_point_set = False

        self.load_state()

    def save_state(self):
        mkdir_if_exist_not('{}'.format(self.args.output_dir))
        log.info('saving data to {}'.format(self.filepath))
        with open(self.filepath, 'w') as f:
            f.write(
                json.dumps((
                    self.scale_factor,
                    self.unit_rotation,
                    self.rotation,
                    self.finished,
                    self.line,
                ))
            )
            
    def load_state(self):
        try:
            log.info('loading data from {}'.format(self.filepath))
            with open(self.filepath) as f:
                state = json.loads(f.read())
                (
                    self.scale_factor,
                    self.unit_rotation,
                    self.rotation,
                    self.finished,
                    self.line) = state
                
        except IOError:
            log.exception('====')
            #raise IOError
        except:
            log.exception('====')
            
    def imshow(self, name, img):
        imshow(name, img, 1/self.scale_factor)

    def is_finished(self):
        return self.finished

    def draw_line(self, img, line, color=(255,0,0), thickness=4):
        (x1, y1), (x2, y2) = line            
        px, py, qx, qy = extend_line_to_boundary(img, line)
        
        cv2.line(img,
                 (px, py), (qx, qy),
                 COLOR_LINE,
                 LINE_THICKNESS)
        
        cv2.circle(img, tuple(line[0]), CIRCLE_RADIUS, COLOR_CIRCLE, -1)
        cv2.circle(img, tuple(line[1]), CIRCLE_RADIUS, COLOR_CIRCLE, -1)

        cv2.line(img,
                 (x1, y1), (x2, y2),
                 COLOR_LINE_SEGMENT,
                 LINE_SEGMENT_THICKNESS)

    def draw(self):
        del self.img
        self.img = self.source.copy()
        self.draw_line(self.img, self.line)

    def callback(self, event, x, y, flags, param):
        temp = [self.scale_factor * x, self.scale_factor * y]
        
        if event == cv2.EVENT_LBUTTONDOWN:
            log.info('1 temp: {}'.format(pformat(temp)))
            self.first_point = temp
            log.info('1 line: {}'.format(pformat(self.line)))
            self.line[0] = temp
            self.first_point_set = True
            self.draw()
            
        if event == cv2.EVENT_MOUSEMOVE:
            if self.first_point_set:
                log.info('2 temp: {}'.format(pformat(temp)))
                self.line[1] = temp
                log.info('2 line: {}'.format(pformat(self.line)))
                self.draw()
        
        if event == cv2.EVENT_LBUTTONUP:
            log.info('3 temp: {}'.format(pformat(temp)))
            if self.first_point_set:
                self.line[1] = temp
                self.first_point_set = False
                log.info('3 line: {}'.format(pformat(self.line)))
                self.draw()
        
    def event_loop(self):
        print('image state == {} and args.force == {}'.format(self.finished, self.args.force))

        if self.finished:
            print('already tagged image!!!')
        
        if self.finished == False or self.args.force:
            self.finished = False
            self.draw()

            self.source = rotate(self.source_backup, self.rotation)
            
            while True:
                """
                self.img = self.source.copy()
                x1 = random.randint(0, self.img.shape[1])
                y1 = random.randint(0, self.img.shape[0])
                x2 = random.randint(0, self.img.shape[1])
                y2 = random.randint(0, self.img.shape[0])
                cv2.line(self. img,
                         (x1, y1), (x2, y2),
                         (0, 255, 0),
                         5)
                """
                
                cv2.setMouseCallback(self.name, self.callback)
                self.imshow(self.name, self.img)

                k = cv2.waitKey(100) & 0xFF
                if k == 27:
                    log.debug('setting state to grab')
                    if self.line_backup != None:
                        self.line = self.line_backup

                elif k == ord('q'):
                    log.debug('quit!')
                    exit(0)

                elif k == ord('R'):
                    log.debug('rotate image {} degree'.format(self.rotation))
                    self.source = rotate(self.source, self.unit_rotation)
                    self.rotation += self.unit_rotation
                    self.draw()

                elif k == ord('r'):
                    log.debug('rotate image {} degree'.format(-self.rotation))
                    self.source = rotate(self.source, -self.unit_rotation)
                    self.rotation -= self.unit_rotation
                    self.draw()

                elif k == ord('c'):
                    log.debug('clear rotation')
                    self.source = self.source_backup.copy()
                    self.rotation = 0
                    self.draw()                

                elif k == ord('f'):
                    log.debug('rotate image {} degree'.format(-180))
                    self.source = rotate(self.source, -180)
                    self.rotation -= 180
                    self.draw()

                elif k == ord(' '):
                    self.save_state()

                elif k == ord('F'):
                    log.info('setting image status to be finished')
                    self.finished = True
                    self.save_state()

                elif k == 13:
                    self.save_state()
                    break

            self.save_state()

        cv2.destroyWindow(self.name)
        return self.line

def process(args):

    image = cv2.imread('0000.png', -1)
    imshow('temp 1', image)
    cv2.waitKey(0)
    vetti = Vetti(args, os.path.basename(args.filepath), image)
    line = vetti.event_loop()

    mask  = np.ones(image.shape, dtype=np.uint8)
    imshow('temp 2', mask)
    cv2.waitKey(0)
    roi_corners = np.array(
        [[ (10, 10), (300, 300), (10, 300), (240, 300)]],
        dtype=np.int32)

    n_channels = image.shape[2]
    ignore_mask_color = (255, ) * n_channels
    
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)

    imshow('temp 3', masked_image)
    cv2.waitKey(0)
    
    right_mask = np.ones(image.shape, dtype=np.uint8)
           
import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Thaal-Vetti')


    parser.add_argument('-s','--size',
                        help='size of the resulting shape',
                        default=120, dest='size')
    
    parser.add_argument('-v', '--verbose',
                        help='shows all the grid overlayed in input image',
                        action='store_true', default=False, dest='verbose')

    parser.add_argument('-V', '--very-verbose',
                        help='shows all the pieces of the characters',
                        action='store_true', default=False, dest='very_verbose')
        
    parser.add_argument('-F','--force',
                        help='start tagging even if finished set to true',
                        action='store_true', default=False, dest='force')

    parser.add_argument('-i','--input-dir',
                        help='path to the image file',
                        default='pages', dest='input_dir')

    parser.add_argument('-o','--output-dir',
                        help='path to the image file',
                        default='split_pages', dest='output_dir')
    
    args = parser.parse_args()

    pprint(args)
    
    args.filepath = 'dummy'
    process(args)
    
    """
    for filepath in glob('{}/*.jpg'.format(args.input_dir)):
        log.info('processing {}'.format(filepath))
        args.filepath = filepath
        total_count, shapes = process(args)
        print('total count: {}'.format(total_count))
        
        write_shapes(args, shapes)
        cv2.destroyAllWindows()
        
    """
