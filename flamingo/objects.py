
# coding: utf-8

# In[310]:

import os
import cv2
import uuid
import time
import zlib
import hashlib
import cPickle as pickle


# In[394]:

class CompressedFlamingoObject(object):
    
    
    _data = None
    
    
    def __init__(self, data=None):
        if data:
            self.compress(data)
        
        
    def __repr__(self):
        s = 'Compressed Flamingo Object:\n'
        if self._data:
            m = hashlib.md5()
            m.update(self._data)
            s += '  hash : %s\n' % m.hexdigest()
            s += '  byte : %d\n' % len(self._data)
        else:
            s += '  <empty>'
        return s
    

    def compress(self, data):
        if hasattr(data, '_ext'):
            self._ext = data._ext
        self._data = zlib.compress(pickle.dumps(data), 9)
        return self


    def decompress(self):
        return pickle.loads(zlib.decompress(self._data))
    
    
    def dump(self, fp):
        fp.write(self._data)

            
    def load(self, fp):
        self._data = fp.read()
        return self
    
    
    @property
    def data(self):
        return self._data


class FlamingoObjectProperty(object):
    
    
    _forhash = None
    _value = None
    
    
    def __init__(self):
        pass
    
    
    def __repr__(self):
        s = 'Flamingo Object Property:\n'
        if self._value:
            s += '  hash : %s\n' % self._forhash
            s += '  type : %s\n' % type(self._value).__name__
        else:
            s += '  <empty>'
        return s
    
    
    @property
    def value(self):
        return self._value
    
    
    @value.setter
    def value(self, args):
        if len(args) < 2:
            raise ValueError('No hash and value supplied, use prop = hash, value syntax')
        self._forhash = args[0]
        self._value = args[1]
    

class FlamingoObject(object):
    
    _type = 'Flamingo Object'
    _ext = '.fo'
    _empty = True
    
    _hash = ''
    #_tstamp = 0.
    _config = ''
    
    config = {}
    
    
    def __init__(self, filename_or_obj=None, config=None):
        
        if config:
            self.load_config(config)
            
        if filename_or_obj:
            if isinstance(filename_or_obj, FlamingoObject):
                self.copy_attributes(filename_or_obj)
            elif isinstance(filename_or_obj, str):
                self.load(filename_or_obj)
            else:
                raise TypeError('Expect str or FlamingoObject, %s found' % type(filename_or_obj).__name__)
                
        self.update_hash()
    
    
    def __repr__(self):
        s = '%s:\n' % self._type
        if self._empty:
            s += '  <empty>\n'
        else:
            s += '  hash : %s\n' % self._hash
            #s += '  time : %s\n' % time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(self._tstamp))
        return s
    
    
    def dump(self, filename):
        '''Dump object as compressed pickle file'''
        
        filename = ''.join((os.path.splitext(filename)[0], self._ext))
        with open(filename, 'wb') as fp:
            CompressedFlamingoObject(self).dump(fp)

            
    def load(self, filename):
        '''Load object attributes from compressed pickle file'''
        
        with open(filename, 'rb') as fp:
            try:
                obj = CompressedFlamingoObject().load(fp).decompress()
            except:
                raise IOError('Invalid %s file' % self._type)
        self.copy_attributes(obj)
                
                
    def load_config(self, filename):
        '''Load Flamingo configuration file'''
        
        with open(filename, 'r') as fp:
            self.config = json.load(fp)
            self._config = os.path.abspath(filename)
            
        self.update_hash()
            
            
    def copy_attributes(self, obj):

        if self._type != obj._type:
            raise IOError('Invalid %s object, this seems to be a %s object' % (self._type, obj._type))
            
        for name, attr in obj.iterate_attributes():
            setattr(self, name, attr)
            
        self.update_hash()

            
    def iterate_attributes(self):
        for name in dir(self):
            if name.startswith('__'):
                continue
            attr = getattr(self, name)
            if hasattr(attr, '__call__'):
                continue
            yield name, attr
                
            
    def update_hash(self):
        self._empty = False
        #self._tstamp = None
        self._hash = None
        m = hashlib.md5()
        m.update(CompressedFlamingoObject(self).data)
        #self._tstamp = time.time()
        self._hash = m.hexdigest()
    
    
class FlamingoImageObject(FlamingoObject):
    
    _type = 'Flamingo Image Object'
    _ext = '.fio'
    
    _name = None
    
    image = None
    channels = None
    segmentation = None
    features = None
    annotation = None
    prediction = None
    
    
    def __init__(self, image_or_obj, config=None):
        try:
            super(FlamingoImageObject, self).__init__(filename_or_obj=image_or_obj, config=config)
        except (IOError, TypeError):
            self._name = os.path.split(image_or_obj)[1]
            self.imread(image_or_obj)
        self.update_hash()
            
            
    def __repr__(self):
        s = super(FlamingoImageObject, self).__repr__()
        s += '  name : %s\n' % self._name
        return s
    
    
    def extract_channels(self):
        pass
    
    
    def create_segmentation(self):
        pass
    
    
    def extract_features(self):
        pass
    
    
    def add_annotation(self):
        pass
    
    
    def make_prediction(self):
        pass
    
    
    def imread(self, filename):
        self.image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
        self.update_hash()
        
        
    def imwrite(self, filename):
        cv2.imwrite(filename, cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR))
        
        
    def imshow(self):
        fig, axs = plt.subplots()
        axs.imshow(self.image)
        return axs

    
class FlamingoDatasetObject(FlamingoObject):
    
    _type = 'Flamingo Dataset Object'
    _ext = '.fdo'
    
    images = []
    statistics = []
    partitions = []
        
    
    def __init__(self, images_or_obj=None, config=None):
        
        try:
            super(FlamingoDatasetObject, self).__init__(filename_or_obj=images_or_obj, config=config)
        except (IOError, TypeError):
            self.add_images(images_or_obj)
            
        self.update_hash()
        
        
    def __repr__(self):
        s = super(FlamingoDatasetObject, self).__repr__()
        if len(self.images) > 0:
            s += '\n'
            for image in self.images:
                s += '  %s\n' % image._name
        return s

                
    def add_images(self, images):
        if isinstance(images, list):
            for image in images:
                self.add_image(image)
        elif isinstance(images, str):
            for image in glob.glob(images):
                self.add_image(image)
        else:
            raise ValueError('Definition of images not understood')
            
            
    def add_image(self, image):
        obj = FlamingoImageObject(image)
        self.images.append(obj)
        self.update_hash()
        
        
    def compute_statistics(self):
        pass
    
    
    def partition(self):
        pass
    
        
class FlamingoModelObject(FlamingoObject):
    
    _type = 'Flamingo Model Object'
    _ext = '.fmo'
    
    dataset = None
    models = []
    
    
    def __init__(self, dataset_or_obj=None, config=None):
        
        if isinstance(dataset_or_obj, FlamingoDatasetObject):
            super(FlamingoModelObject, self).__init__(config=config)
            self.set_dataset(dataset_or_obj)
        else:
            try:
                super(FlamingoModelObject, self).__init__(filename_or_obj=dataset_or_obj, config=config)
            except (IOError, TypeError):
                self.set_dataset(dataset_or_obj)
                
        self.update_hash()
                

    def set_dataset(self, dataset):
        if not isinstance(dataset, FlamingoDatasetObject):
            dataset = FlamingoDatasetObject(dataset)
        self.dataset = dataset
        self.update_hash()
    
    
    def train(self):
        pass

    
    def test(self):
        pass


    def validate(self):
        pass

    
    def regularize(self):
        pass


# In[395]:

FIO1 = FlamingoImageObject('/Users/hoonhout/Projects/11200125.008-highspeedsg/output/timestacks_smooth/_selection/b3w0t101_No=0001_t=0266370ms_000000346733_000001_000002.png')
FIO2 = FlamingoImageObject('/Users/hoonhout/Projects/11200125.008-highspeedsg/output/timestacks_smooth/_selection/b3w0t101_No=0001_t=0266370ms_000000346733_000001_000002.png')
FIO3 = FlamingoImageObject('/Users/hoonhout/Projects/11200125.008-highspeedsg/output/timestacks_smooth/_selection/b3w0t101_No=0001_t=0266370ms_000000346733_000001_000002.png')
FIO1.dump('test.fio')

FDO = FlamingoDatasetObject([FIO1,FIO2,FIO3])
FDO.dump('test.fdo')

FMO = FlamingoModelObject(FDO)
FMO.dump('test.fmo')

print FIO1
print FIO2
print FDO
print FMO


# In[396]:

FIO1 = FlamingoImageObject('/Users/hoonhout/Projects/11200125.008-highspeedsg/output/timestacks_smooth/_selection/b3w0t101_No=0001_t=0266370ms_000000346733_000001_000002.png')
CFO = CompressedFlamingoObject(FIO1)
print CFO

CFO = CompressedFlamingoObject()
CFO


# In[397]:

FMO2 = FlamingoModelObject()
FMO2.load('test.fmo')
print FMO2

FMO2.update_hash()
print FMO2

FMO2.set_dataset(FDO)
print FMO2

FMO2.set_dataset(FlamingoDatasetObject(FDO))
print FMO2


# In[400]:




# In[322]:

CompressedFlamingoObject(FMO2)


# In[367]:

FOP = FlamingoObjectProperty()
FOP.value = 'test', 3
FOP


# In[363]:




# In[ ]:



