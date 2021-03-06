
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

import pdb
    
import logging
FORMAT_STRING = "%(levelname)-8s:%(name)-8s.%(funcName)-8s>> %(message)s"

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

from pprint import pprint, pformat

SLOPE_INFO_POSITION  = ( 10, 50)
POINTS_INFO_POSITION = (200, 50)

COLOR_LINE         = (212, 188,   0)
COLOR_LINE_SEGMENT = (000,  44, 221)
COLOR_CIRCLE       = (34,   87, 255)

CIRCLE_RADIUS = 24
LINE_SEGMENT_THICKNESS = 12
LINE_THICKNESS = 3


def mkdir_if_exist_not(name):
    if not os.path.isdir(name):
        return os.makedirs(name)

def imshow(name, img, resize_factor = 0.4):
    ret = cv2.imshow(name,
                     cv2.resize(img,
                                (0, 0),
                                fx=resize_factor,
                                fy=resize_factor))
    
    #cv2.moveWindow(name, 100, 100)
    
    return ret
    
def put_text(img, text, position):
    cv2.putText(img,
                text,
                position,
                cv2.FONT_HERSHEY_SIMPLEX,  
                1,
                COLOR_LINE,
                3,
                cv2.LINE_AA)
        
def slope(x1, y1,  x2, y2):
    if x2 != x1:
        return( (y2 - y1) / (x2 - x1) )
    else:
        return 'NA'


def intercept(x1, y1, x2, y2):
    m = slope(x1, y1, x2, y2)
    #y = mx + b
    if m != 'NA':
        assert int(y1 - m * x1) == int(y2 - m * x2), \
            '{} =/= {} are you sure, this equation is correct?'.format(
                int(y1 - m * x1),
                int(y2 - m * x2))
        
        return (y1 - m * x1)
    else:
        return y1

def extend_line_to_boundary(img, line):
    (x1, y1), (x2, y2) = line

    log.debug(
        '(x1, y1), (x2, y2): ({}, {}), ({}, {})'.format(
            x1, y1, x2, y2))
    
    m = slope(x1, y1, x2, y2)
    log.debug('slope: {}'.format(m))

    h, w = img.shape[:2]
    log.debug('h, w: {}, {}'.format(h, w))
    
    if m != 'NA':
        px, py = 0, -(x1 - 0) * m + y1
        qx, qy = w, -(x2 - w) * m + y2
        
    else:
        px , py = x1, 0
        qx , qy = x1, h

    px, py, qx, qy = [ int(i) for i in [px, py, qx, qy] ]
    log.debug('px, py, qx, qy: {}, {}, {}, {}'.format(px, py, qx, qy))

    return px, py, qx, qy

def extend_line_to_boundary3(img, line):
    """
    find slope and intercept and then use that to find the intersection
    """
    h, w, _ = img.shape

    log.debug('h, w: {}, {}'.format(h , w))

    (x1, y1), (x2, y2) = line

    log.debug(
        '(x1, y1), (x2, y2): ({}, {}), ({}, {})'.format(
            x1, y1, x2, y2))


    m, b = slope(x1, y1, x2, y2), intercept(x1, y1, x2, y2)

    log.debug('m, b: {}, {}'.format(m, b))

    py, qy = 0, h

    if m != 'NA':
        px = (0 - b) / m
        qx = (h - b) / m
    else:
        px = qx = x1

    px, py, qx, qy = [ int(i) for i in [px, py, qx, qy] ]
    log.debug('px, py, qx, qy: {}, {}, {}, {}'.format(px, py, qx, qy))

    return px, py, qx, qy

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



def gamma_correct(image, gamma=1.0):
    gamma_inv = 1.0 / gamma
    table = np.array( 255 * [ (i / 255.0) ** gamma_inv  for i in np.arange(0, 256) ],
                      dtype=np.uint8,
    )

    return cv2.LUT(image, table)

class Vetti:
    STATE_LINE_DRAWING = -1
    
    def __init__(self, args, name, img,
                 scale_factor = 0.5,
                 unit_rotation = 0.1):

        self.args = args
        self.name = name
        self.filepath = 'vetti/{}/{}.vetti'.format(self.args.output_dir, self.name)
        
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

        h, w, *_ = self.img.shape
        self.line = [[w//2, 0], [w//2, h]]
        log.debug('default-line: {}'.format(pformat(self.line)))

        self.first_point = None
        self.first_point_set = False

        if args.clear:
            self.save_state()
        else:
            self.load_state()

    def save_state(self):
        mkdir_if_exist_not('vetti/{}'.format(self.args.output_dir))
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
            log.info('laod_state() IOError ignored')
            #raise IOError
        except:
            log.exception('====')

    def title_message(self, name=None):
        count, total = self.name.split(os.environ['DELIMITER'])
        total = total.split('.')[0]
        
        return '{} / {}'.format(count, total)
        
    def imshow(self, name, img):
        imshow(name, img, 1/self.scale_factor)

    def is_finished(self):
        return self.finished

    def draw_line(self, img, line, color=(255,0,0), thickness=4):
        (x1, y1), (x2, y2) = line
        h, w, c = img.shape

        px, py, qx, qy = extend_line_to_boundary3(img, line)

        cv2.line(img,
                 (px, py), (qx, qy),
                 COLOR_LINE,
                 LINE_THICKNESS)
        
        cv2.circle(img,
                   tuple(line[0]),
                   CIRCLE_RADIUS,
                   COLOR_CIRCLE,
                   -1)
        
        cv2.circle(img,
                   tuple(line[1]),
                   CIRCLE_RADIUS,
                   COLOR_CIRCLE,
                   -1)

        cv2.line(img,
                 (x1, y1), (x2, y2),
                 COLOR_LINE_SEGMENT,
                 LINE_SEGMENT_THICKNESS)

        put_text(self.img,
                 '({})'.format(
                     str(slope(x1, y1, x2, y2))[:5]
                 ),
                 SLOPE_INFO_POSITION)        
        
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
            put_text(self.img,
                     '({}, {}) - ({}, {})'.format(*self.line[0],
                                                  *temp),
                     POINTS_INFO_POSITION)
        
        if event == cv2.EVENT_LBUTTONUP:
            log.info('3 temp: {}'.format(pformat(temp)))
            if self.first_point_set:
                self.line[1] = temp
                self.first_point_set = False
                log.info('3 line: {}'.format(pformat(self.line)))
                self.draw()
        
    def event_loop(self):
        print('image state == {} and args.force == {}'.format(self.finished, self.args.force))

        retval = 'DONE'
        
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
                self.imshow(self.title_message(), self.img)

                k = cv2.waitKey(100) & 0xFF

                if k == 27:
                    log.debug('setting state to grab')
                    if self.line_backup != None:
                        self.line = self.line_backup

                elif k == ord('q'):
                    log.info('quit!')
                    retval = 'BREAK';
                    break

                elif k == ord('s'):
                    log.info('single page doc')
                    retval = 'DONOTHING';
                    break

                elif k == ord(' '):
                    self.save_state()

                elif k == ord('F'):
                    log.info('setting image status to be finished')
                    self.finished = True
                    self.save_state()
                    
                elif k == 83:
                    log.debug('moving right by 1')
                    self.line[0][0] += 1
                    self.line[1][0] += 1
                    
                    self.draw()
                    
                elif k == 81:
                    log.debug('moving left by 1')
                    self.line[0][0] -= 1
                    self.line[1][0] -= 1

                    self.draw()
                
                    
                elif k == 13:
                    self.save_state()
                    break

            self.save_state()

        cv2.destroyWindow(self.title_message())
        return retval, self.line

def callback_show_point_func(img):
    def callback(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            log.debug('({}, {})'.format(x, y))
            put_text(img,
                     '({}, {})'.format(x, y),
                     (100, 150))

    return callback
    
def process(args):

    image = cv2.imread(args.filepath, -1)

    if args.verbose:
        imshow('temp 1', image)
        cv2.waitKey(0)
        
    vetti = Vetti(args,
                  os.path.basename(args.filepath),
                  image,
                  scale_factor=args.scale_factor)
    
    retval, line = vetti.event_loop()

    if retval == 'DONE':

        print(image.shape)

        h, w, n_channels = image.shape
        x1, y1, x2, y2   = extend_line_to_boundary3(image, line)
        #(x1, y1), (x2, y2)= line

        log.debug('(x1, y1) (x2, y2), h, w: {}'.format(
            pformat( [[x1, y1], [x2, y2], h, w] )
        ))

        left_roi  = [[(0, 0), (0, h), (x2, y2), (x1, y1),]]
        right_roi = [[(w, 0), (w, h), (x2, y2), (x1, y1),]]

        log.debug('left roi: '.format(pformat(left_roi)))
        log.debug('right roi: '.format(pformat(right_roi)))

        def mask(roi):
            mask  = np.zeros(image.shape, dtype=np.uint8)
            roi_corners = np.array(roi, dtype=np.int32)
            ignore_mask_color = (255, ) * n_channels

            cv2.fillPoly(mask, roi_corners, ignore_mask_color)
            masked_image = cv2.bitwise_and(image, mask)
            #np.copyto(masked_image, image, mask)
            #masked_image += image * (mask > 0)

            masked_image += 255 * np.ones(image.shape, dtype=np.uint8) * (mask == 0)

            log.debug('mask shape: {}'.format(pformat(mask.shape)))

            if args.debug:
                cv2.polylines(masked_image,
                              roi_corners,
                              False,
                              COLOR_LINE_SEGMENT,
                              5)

                #--- padding
                # ht, wd, cc = masked_image.shape
                # hh, ww, = h + 200, w + 200

                # xx = (ww - wd) // 2
                # yy = (hh - ht) // 2

                # temp = masked_image
                # masked_image = np.full( (hh,ww,cc),
                #                         (255, 255, 255, 255),
                #                         dtype=np.uint8)

                # masked_image[yy:yy+ht, xx:xx+wd] = temp
                #---

                put_text(masked_image,
                         '1 ({}, {})'.format(*roi[0][0]),
                         tuple(i + 100 for i in roi[0][0]))

                put_text(masked_image,
                         '2 ({}, {})'.format(*roi[0][1]),
                         tuple(i + 100 for i in roi[0][1]))

                put_text(masked_image,
                         '3 ({}, {})'.format(*roi[0][2]),
                         (w//2, 100))

                put_text(masked_image,
                         '4 ({}, {})'.format(*roi[0][3]),
                         (w//2, h - 100 ))

            return masked_image

        left  = mask(left_roi)
        right = mask(right_roi)

        xlim = max(x1, x2)
        left = left[:, :xlim]

        xlim = min(x1, x2)
        right = right[:, xlim:]

        if args.debug:
            imshow('left', left)
            imshow('right', right)
            cv2.waitKey(0)


        output_path = '{}{}{}_0.png'.format(
                args.output_dir,
                os.path.sep,
                os.path.splitext(
                    os.path.basename(args.filepath)
                )[0],
            )
        log.info('writing left side to {}'.format(output_path))

        cv2.imwrite(output_path, left)

        output_path = '{}{}{}_1.png'.format(
                args.output_dir,
                os.path.sep,
                os.path.splitext(
                    os.path.basename(args.filepath)
                )[0],
            )
        log.info('writing right side to {}'.format(output_path))

        cv2.imwrite(output_path, right)
        
    if retval == 'DONOTHING':
        output_path = '{}{}{}_0.png'.format(
                args.output_dir,
                os.path.sep,
                os.path.splitext(
                    os.path.basename(args.filepath)
                )[0],
            )
        log.info('writing iamge side to {}'.format(output_path))

        cv2.imwrite(output_path, image)
        
    return retval

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Thaal-Vetti')


    parser.add_argument('-s','--size',
                        help='size of the resulting shape',
                        default=120, dest='size')
    
    parser.add_argument('-v', '--verbose',
                        help='shows all the grid overlayed in input image',
                        action='store_true', default=False, dest='verbose')

    parser.add_argument('--display-resolution',
                        help='display resolution of input image',
                        action='store', default=0.3,
                        dest='scale_factor', type=float)

    parser.add_argument('-C','--clear',
                        help='start tagging fresh',
                        action='store_true', default=False, dest='clear')
    
    parser.add_argument('-D', '--debug',
                        help='shows all the pieces of the characters',
                        action='store_true', default=False, dest='debug')
        
    parser.add_argument('-F','--force',
                        help='start tagging even if finished set to true',
                        action='store_true', default=False, dest='force')

    parser.add_argument('-i','--input',
                        help='path to the image file',
                        default='pages', dest='filepath')

    parser.add_argument('-o','--output-dir',
                        help='path to the image file',
                        default='split_pages', dest='output_dir')
    
    
    args = parser.parse_args()

    pprint(args)
    
    #args.filepath = 'data/0000.png'
    #process(args)

    if args.debug:
        args.verbose = True
        
    if args.verbose:

        logging.basicConfig(format=FORMAT_STRING)
        log = logging.getLogger(__name__)
        log.setLevel(logging.DEBUG)

    mkdir_if_exist_not(args.output_dir)    
    
    errored_pages = []
    retval = process(args)
    cv2.destroyAllWindows()
        

