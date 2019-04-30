import numpy as np
from fastai.vision import *

# ItemBase subclass
class MRNetCase(ItemBase):

  # ItemBase.data   # this needs to be developed in parallel with the ItemList's .get method
  # imagine that .get will return a tuple of three np.arrays, an erray for each plane
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
  # class variables
    # _bunch
    # _processor
    # _label_cls

  # __init__ arguments

  # core methods
    # get
    # reconstruct
    # analyze_pred

  # advanced show methods
    # show_xys
    # show_xyzs