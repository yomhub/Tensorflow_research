import os,sys
import tarfile
import json
import math
import numpy as np
import tensorflow as tf

_LOCAL_DIR = os.path.split(__file__)[0]
_ANN_FILENAME = 'ctw-annotations.tar.gz'
_STW_DIR = os.path.join(_LOCAL_DIR, 'ctw')
_TRAIN_DIR = os.path.join(_LOCAL_DIR, 'ctw','train')
_TEST_DIR = os.path.join(_LOCAL_DIR, 'ctw','test')
_LOG_FILE = os.path.join(_LOCAL_DIR, 'ctw','log.txt')
_TOTAL_TRAIN_NUM = 1000 * 25 + 887
_TOTAL_TEST_NUM = 1000 * 6 + 398

def reset(tarinfo):
    tarinfo.name = os.path.basename(tarinfo.name)
    tarinfo.uid = tarinfo.gid = 0
    tarinfo.uname = tarinfo.gname = 'root'
    tarinfo.mtime = 0
    tarinfo.mode = 0o644
    return tarinfo

def _load_and_preprocess_image(imgdir, outsize=None):
    fd=""
    if isinstance(imgdir,str):
        fd=imgdir
    else:
        fd=str(imgdir.numpy())
    image = tf.image.decode_image(tf.io.read_file(fd))
    if(outsize!=None):
        image = tf.image.resize(image, outsize)
    # image = tf.reshape(image,(1, image.shape[0], image.shape[1], image.shape[2]))
    image = tf.dtypes.cast(image, tf.uint8)

    return image

def _ratio(pbx):
    upratio=math.atan((pbx[1][1]-pbx[0][1])/
    (pbx[1][0]-pbx[0][0]))
    downratio=math.atan((pbx[3][1]-pbx[2][1])/
    (pbx[3][0]-pbx[2][0]))
    return (upratio + downratio)/2

def _read_jsl(jsl_dir):
    with open(jsl_dir, 'r') as jesl:
        json_list = jesl.read().split('\n')
        rt_list = []
        for json_str in json_list:
            if(json_str != ''):
                rt_list.append(json.loads(json_str))
    return rt_list

def _read_json( ctw_dir = _STW_DIR,
                ann_filename = _ANN_FILENAME.split('.')[0]
    ):
    datas_dir = ctw_dir
    ann_dir = os.path.join(datas_dir, ann_filename)
    if(not(os.path.exists(ann_dir)) or len(os.listdir(ann_dir)) != 4):
        with tarfile.open(os.path.join(datas_dir, _ANN_FILENAME), 'r|gz') as tar:
            tar.extractall(path=ann_dir)
            if(len(tar.getmembers()) != 4):
                return []

    with open(os.path.join(ann_dir, 'info.json'), 'r') as info_jetson:
        info_list = json.load(info_jetson)

    test_list = _read_jsl(os.path.join(ann_dir, 'test_cls.jsonl'))
    train_list = _read_jsl(os.path.join(ann_dir, 'train.jsonl'))
    val_list = _read_jsl(os.path.join(ann_dir, 'val.jsonl'))
    train_list.append(val_list)
    return info_list, test_list, train_list

class CTW():
    def __init__(
        self,
        out_format = "DICT",
        box_format = "ADJB",
        cls_type = "BOOL",
        out_size = None,
        ctw_dir = _STW_DIR,
        train_img_dir = _TRAIN_DIR,
        test_img_dir = _TEST_DIR,
        ann_filename = _ANN_FILENAME.split('.')[0],
        log_dir = _LOG_FILE,
    ):
        """
        Args: (unavailable yet)
          out_format: "DICT" or "TENSOR"
            DICT: dictionary with member
              "gt_bbox": shape (total_gts, ...)
              "class": shape (total_gts, 1)
            TENSOR: only output single tensor
          box_format: "ADJB", "POLY" or "CLSADJB"
            ADJB: adjusted bbox [xstart, ystart, w, h]
            POLY: polygon [[top left x, top left y],
                          [top right x, top right y],
                          [down right x, down right y],
                          [down left x, down left y],
                          ]
            CLSADJB: [class, xstart, ystart, w, h]
          cls_type: define class value, "UNICODE" or "BOOL"
            UNICODE: return text unicode value
            BOOL: return 1 if obj is test, other case 0 (tf.int8)
          out_size: [width,height] or None means orign size
        """
        # load info list in ctw_dir/ann_filename/ floder
        
        # _info_list['test_cls','test_det','train','val']: a list of dataset
        # _info_list[*][0]['file_name','image_id','height','width']

        # test_list[0]['file_name','image_id','height','width']
        # test_list[0]['proposals']
        # test_list[0]['proposals'][0]: num of single text in word[0] or word in picture[0]
        # test_list[0]['proposals'][0]['adjusted_bbox']: [start, ystart, w, h]
        # test_list[0]['proposals'][0]['polygon']: list of box 4 points
        #                                         top left:[x, y]
        #                                         top right:[x, y]
        #                                         down right:[x, y]
        #                                         down left:[x, y]

        # _train_list[0]['annotations']: Contains N words in the picture 
        # _train_list[0]['ignore']: Contains N ignore words in the picture 
        # _train_list[0]['file_name']: string, name of file like '0000172.jpg'
        # _train_list[0]['image_id']: string, name of file like '0000172'
        # _train_list[0]['height']: int, height of image
        # _train_list[0]['width']: int, width of image

        # len(_train_list[0]['annotations']): num of words in picture[0]
        # len(_train_list[0]['annotations'][0]): num of single text in word[0] in picture[0]
        # text[0] in word[0] in picture[0]
        # _train_list[0]['annotations'][0][0]['adjusted_bbox']: [xstart, ystart, w, h]
        # _train_list[0]['annotations'][0][0]['attributes']: list of string, may have "bgcomplex", "distorted", "raised"
        # _train_list[0]['annotations'][0][0]['is_chinese']: true or flase
        # _train_list[0]['annotations'][0][0]['polygon']: list of box 4 points
        #                                                 top left:[x, y]
        #                                                 top right:[x, y]
        #                                                 down right:[x, y]
        #                                                 down left:[x, y]
        # _train_list[0]['annotations'][0][0]['text']: string, unicode in chinese

        # _val_list: same as _train_list
        self._info_list, self._test_list, self._train_list = _read_json(ctw_dir,ann_filename)
        self._init_conter = 0
        self._train_dir = train_img_dir
        self._test_dir = test_img_dir
        self._log_dir = log_dir
        self._log = open(self._log_dir,'w')
        self._out_size = out_size
        if(out_format=="DICT"):
          self.out_format = out_format
        else:
          self.out_format = "TENSOR"

        if(box_format=="ADJB"):
          self.box_format = box_format
        elif(box_format=="POLY"):
          self.box_format = "POLY"
        else:
          self.box_format = "CLSADJB"

        if(cls_type=="UNICODE"):
          self.cls_type = cls_type
        else:
          self.cls_type = "BOOL"

    def __del__(self):
        self._log.close()

    def reset(self):
        self._init_conter = 0
        self._log.close()
        self._log = open(self._log_dir,'w')
    
    def setconter(self,cont):
        self._init_conter = cont if len(self._train_list) > cont else 0

    def _format_output(self,out_annotation):
      return [int(out_annotation['is_chinese']),]+out_annotation["adjusted_bbox"]

    def read_batch(self, batch_size=50):
        conter = batch_size if batch_size > 0 else 10
        batch_size = batch_size if batch_size > 0 else 10
        i=self._init_conter
        y_list=[]
        img_arr_list=[]
        if(self._out_size!=None):
          ratioY = self._out_size[0]/self._train_list[0]['height']
          ratioX = self._out_size[1]/self._train_list[0]['width']
        while conter > 0:
            ytmp=[]
            try:
                tmp=_load_and_preprocess_image(
                    os.path.join(self._train_dir, self._train_list[i]['file_name']),
                    self._out_size)
                conter-=1
                # for stack
                if(len(tmp.shape)==4):
                    tmp=tf.reshape(tmp[0,:,:,:],[tmp.shape[1],tmp.shape[2],tmp.shape[3]])
                img_arr_list.append(tmp)
                    
                for word in self._train_list[i]['annotations']:
                    for text in word:
                        if(self._out_size!=None):
                          gtmp=text["adjusted_bbox"]
                          gtmp=[gtmp[0]*ratioX,gtmp[1]*ratioY,gtmp[2]*ratioX,gtmp[3]*ratioY]
                          ytmp.append([float(text['is_chinese']),]+gtmp)
                        else:
                          ytmp.append([float(text['is_chinese']),]+text["adjusted_bbox"])
                y_list.append(tf.convert_to_tensor(ytmp))
                
            except IndexError:
                return None
            except:
                self._log.write("Can't read image "+ os.path.join(self._train_dir, self._train_list[i]['file_name']) +"\n")
            i+=1

        self._init_conter=i
        img_arr_list = tf.stack(img_arr_list)
        if(len(y_list)==1):
          y_list = y_list[0]
        return img_arr_list,y_list

    def pipline_entry(self):
        if(self._out_size!=None):
          ratioY = self._out_size[0]/self._train_list[0]['height']
          ratioX = self._out_size[1]/self._train_list[0]['width']
        while True:
            try:
                ytmp=[]
                xtmp=_load_and_preprocess_image(
                    os.path.join(self._train_dir, self._train_list[self._init_conter]['file_name']),
                    self._out_size)
                if(len(xtmp.shape)==3):
                    xtmp=tf.reshape(xtmp,[1,xtmp.shape[0],xtmp.shape[1],xtmp.shape[2]])
                for word in self._train_list[self._init_conter]['annotations']:
                    for text in word:
                        if(self._out_size!=None):
                          gtmp=text["adjusted_bbox"]
                          gtmp=[
                            gtmp[0]*ratioX if gtmp[0]>=0.0 else 0,
                            gtmp[1]*ratioY if gtmp[1]>=0.0 else 0,
                            gtmp[2]*ratioX if (gtmp[2]+gtmp[0])<=self._train_list[0]['width'] else (self._train_list[0]['width']-gtmp[2]-gtmp[0])*ratioX,
                            gtmp[3]*ratioY if (gtmp[3]+gtmp[1])<=self._train_list[0]['height'] else (self._train_list[0]['width']-gtmp[3]-gtmp[1])*ratioY,
                          ]
                          ytmp.append([float(text['is_chinese']),]+gtmp)
                        else:
                          ytmp.append([float(text['is_chinese']),]+text["adjusted_bbox"])
                self._init_conter+=1
                yield xtmp,tf.convert_to_tensor(ytmp,dtype=tf.float32)
            except IndexError:
                self._init_conter = 0
            except:
                self._log.write("Can't read image "+ 
                    os.path.join(self._train_dir, 
                    self._train_list[self._init_conter]['file_name']) +
                    "\n")
                self._init_conter+=1        

