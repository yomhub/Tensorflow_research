import os
import re
# from lxml import etree
from xml.dom import minidom
# import xml.etree.ElementTree as ET
import numpy as np
import tensorflow as tf

_LOCAL_DIR = os.path.split(__file__)[0]
_SVT_DIR = os.path.join(_LOCAL_DIR, 'svt')
_TRAIN_XML = os.path.join(_LOCAL_DIR, 'svt','train.xml')
_TEST_XML = os.path.join(_LOCAL_DIR, 'svt','test.xml')
_LOG_FILE = os.path.join(_LOCAL_DIR, 'svt','log.txt')

def walk(adr):
    imgList = []
    txtList = []
    for root, dirs, files in os.walk(adr):
        for name in files:
            if(os.path.splitext(name)[-1] == ".xml"):
                txtList.append([root, name])
                continue
            if(os.path.splitext(name)[-1] == ".jpg" or
               os.path.splitext(name)[-1] == ".png" or
               os.path.splitext(name)[-1] == ".bmp"
               ):
                imgList.append([root, name])
                continue
    return imgList, txtList


def _load_and_preprocess_image(imgdir,
              sizeX=256,
              sizeY=256):
    fd=""
    if isinstance(imgdir,str):
        fd=imgdir
    else:
        fd=str(imgdir.numpy())
    image = tf.image.decode_image(tf.io.read_file(fd))
    image = tf.image.resize(image, [sizeY,sizeX])
    # image = tf.reshape(image,(1, image.shape[0], image.shape[1], image.shape[2]))
    # image = tf.dtypes.cast(image, tf.uint8)
    return image


def _read_xml(xmldir,
              sizeX=256,
              sizeY=256):
    _x = []
    _y = []
    tmp = []
    xmldoc = minidom.parse(xmldir)
    _count = 0
    tmp=[]
    for itm in xmldoc.getElementsByTagName('image'):
        _x.append(os.path.join(
            os.path.split(xmldir)[0],
            itm.getElementsByTagName("imageName")[0].childNodes[0].nodeValue
        ))
        # tmp.append(list(itm.getElementsByTagName(
        #     "lex")[0].childNodes[0].nodeValue.split(',')))
        lx = int(itm.getElementsByTagName("Resolution")
                 [0].attributes['x'].nodeValue)
        ly = int(itm.getElementsByTagName("Resolution")
                 [0].attributes['y'].nodeValue)

        for trs in itm.getElementsByTagName("taggedRectangles"):
            for tr in trs.getElementsByTagName("taggedRectangle"):
                tmp.append([
                    1.0, # label
                    # start pixel
                    int(tr.attributes["x"].nodeValue)*sizeX/lx,
                    int(tr.attributes["y"].nodeValue)*sizeY/ly,
                    # Rectangle lenth
                    int(tr.attributes["width"].nodeValue)*sizeX/lx,
                    int(tr.attributes["height"].nodeValue)*sizeY/ly,
                    # tr.getElementsByTagName("tag")[0].childNodes[0].nodeValue
                ])
        _y.append(tf.convert_to_tensor(tmp,dtype=tf.float32))
        tmp=[]

    return _x, _y


def _read_swt(swt_dir,
              sizeX=256,
              sizeY=256
              ):
    imgList, txtList = walk(swt_dir)
    _TESTXML = ""
    _TRAINXML = ""
    _xtest = np.array([[[[]]]])
    _ytest = []

    for [r, n] in txtList:
        if(n == "test.xml"):
            _TESTXML = os.path.join(r, n)
            continue
        if(n == "train.xml"):
            _TRAINXML = os.path.join(r, n)
            continue
    if(_TESTXML == "" or _TRAINXML == ""):
        return[]

    _xtdir, _ytest = _read_xml(_TESTXML)
    _xtraindir, _ytrain = _read_xml(_TRAINXML)

    tmp=[]
    for d in _xtraindir:
        tmp.append(_load_and_preprocess_image(d,sizeX,sizeY))
    _xtrain = np.array(tmp)
    tmp=[]
    for d in _xtdir:
        tmp.append(_load_and_preprocess_image(d,sizeX,sizeY))
    _xtest = np.array(tmp)

    return _xtrain,_ytrain,_xtest,_ytest


class SVT():

    def __init__(self,dir,out_size = [512,512]):
        """
            out_size: [y,x]
        """
        self.out_y = out_size[0]
        self.out_x = out_size[1]
        self._xtdir, self._ytest = _read_xml(os.path.join(dir,'test.xml'),self.out_x,self.out_y)
        self._xtraindir, self._ytrain = _read_xml(os.path.join(dir,'train.xml'),self.out_x,self.out_y)
        self._init_conter = 0
        self._init_test_conter = 0

    def read_test_batch(self, batch_size=10):
        mend = min(len(self._xtdir),self._init_test_conter+batch_size)
        inds = slice(self._init_test_conter,mend) 
        readnum = mend-self._init_test_conter
        self._init_test_conter += readnum
        xret = self._xtdir[inds]
        yret = self._ytest[inds]

        if(readnum < batch_size):
            inds = slice(batch_size - readnum)
            self._init_test_conter = batch_size - readnum
            yret += self._ytest[inds]
            xret += self._xtdir[inds]

        img_list = []
        for tname in xret:
            img = _load_and_preprocess_image(tname,self.out_x,self.out_y)
            img_list.append(img)
        img_list = tf.stack(img_list)
        return img_list, yret

    def read_train_batch(self, batch_size=10):
        mend = min(len(self._xtraindir),self._init_conter+batch_size)
        inds = slice(self._init_conter,mend) 
        readnum = mend-self._init_conter
        self._init_conter += readnum
        xret = self._xtraindir[inds]
        yret = self._ytrain[inds]

        if(readnum < batch_size):
            inds = slice(batch_size - readnum)
            self._init_conter = batch_size - readnum
            yret += self._ytrain[inds]
            xret += self._xtraindir[inds]

        img_list = []
        for tname in xret:
            img = _load_and_preprocess_image(tname,self.out_x,self.out_y)
            img_list.append(img)
        img_list = tf.stack(img_list)
        return img_list, yret
        
    def setconter(self,conter):
        self._init_conter = conter

    def caculate_avg(self):
        avg_area_pre_img = []
        for tar in self._ytrain:
            avg_area_pre_img.append(tf.reduce_mean(tar[:,3]*tar[:,4]/float(self.out_y)/float(self.out_x)))
        return tf.convert_to_tensor(avg_area_pre_img)
