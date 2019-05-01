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
  # __str__ representation, or __repr__, since __str__ falls back on __repr__ if not otherwise defined
    def __repr__(self): 
        return f'''
        {self.__class__.__name__} 
        .obj attribute is tuple(axial, coronal, sagittal): 
        {list(e.shape for e in self.obj)}
            
        .data attribute is three middle slices of sagittal
        {self.data.shape}
        {self.data}
        '''

  # apply_tfms (optional)



# ItemList subclass
class MRNetCaseList(ItemList):

  # class variables
    _bunch = MRNetCaseDataBunch
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

    # TODO: reconstruct

    # TODO: analyze_pred

    # TODO: from_df
    @classmethod
    def from_df():
        pass

  # advanced show methods
    # show_xys
    # show_xyzs

class MRNetCaseDataBunch(DataBunch):
    "DataBunch for MRNet knee scan data."

    @classmethod
    def from_df(cls, path:PathOrStr, df:pd.DataFrame, folder:PathOrStr=None, label_delim:str=None, valid_pct:float=0.2,
                fn_col:IntsOrStrs=0, label_col:IntsOrStrs=1, suffix:str='', **kwargs:Any)->'ImageDataBunch':
        "Create DataBunch from a `DataFrame` `df`."
        src = (MRNetCaseList.from_df(df, path=path, folder=folder, suffix=suffix, cols=fn_col)
                .split_by_rand_pct(valid_pct)
                .label_from_df(label_delim=label_delim, cols=label_col))
        return cls.create_from_ll(src, **kwargs)
