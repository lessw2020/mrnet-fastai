import numpy as np
from fastai.vision import *

# ItemBase subclass
class MRNetCase(ItemBase):

  # ItemBase.data   # this needs to be developed in parallel with the ItemList's .get method
  # imagine that .get will return a list of three np.arrays, an array for each plane
    def __init__(self, axial, coronal, sagittal):
        self.axial,self.coronal,self.sagittal = axial,coronal,sagittal
        self.obj = (axial,coronal,sagittal)
        # for .data, initially hard-code taking the middle three slices of the sagittal series
        # middle three grayscale slices, instead of repeating same middle slice 3x
        smid = sagittal.shape[0]//2
        self.data = sagittal[smid-1:smid+2,:,:]
  # __str__ representation
    def __str__(self):
        pass        
  # apply_tfms (optional)



# ItemList subclass
class MRNetCaseList(ItemList):

  # class variables
    # _bunch
    # _processor
    # _label_cls

  # __init__ arguments
  # items for this subclass will likely be a list/iterator of Case strings
  # rather than filenames, since each case has 3 filenames, one for each plane
    def __init__(self, items, **kwargs):
        super().__init__(items, **kwargs)

  # core methods
    # get
    def get(self, i):
        # i indexes self.items, which is a list of Case strings
        case = super().get(i)
        imagearrays = []
        for plane in ('axial','coronal','sagittal'):
            # self.path is available from kwargs of ItemList superclass
            fn  = self.path/plane/(case + '.npy')
            res = self.open(fn)
            imagearrays.append(res)
        assert len(imagearrays) == 3
        return MRNetCase(*imagearrays)

    # since subclassing ItemList rather than ImageList, need an open method
    def open(self, fn): return np.load(fn)

    # reconstruct
    # analyze_pred

  # advanced show methods
    # show_xys
    # show_xyzs