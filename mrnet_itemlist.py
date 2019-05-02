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
        `obj` attribute is tuple(axial, coronal, sagittal): 
        {list(e.shape for e in self.obj)}
            
        `data` attribute is three middle slices of sagittal
        {self.data.shape}
        {self.data}
        '''

  # apply_tfms (optional)

# DataBunch subclass
class MRNetCaseDataBunch(DataBunch):
    "DataBunch for MRNet knee scan data."
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_df(cls, path:PathOrStr, df:pd.DataFrame, folder:PathOrStr=None, label_delim:str=None, valid_pct:float=0.2,
                fn_col:IntsOrStrs=0, label_col:IntsOrStrs=1, suffix:str='', **kwargs:Any)->'ImageDataBunch':
        "Create DataBunch from a `DataFrame` `df`."
        src = (MRNetCaseList.from_df(df, path=path, folder=folder, suffix=suffix, cols=fn_col)
                .split_by_rand_pct(valid_pct)
                .label_from_df(label_delim=label_delim, cols=label_col))
        return cls.create_from_ll(src, **kwargs)


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
        # i indexes self.items, which is an array of filepaths(filenames)
        fn = super().get(i)
        # .stem returns the Case number, as a string
        case = fn.stem
        # also collect whether the case returned with get(i) is in train or valid folder
        tv = 'train' if 'train' in fn.parts else 'valid'
        imagearrays = []
        for plane in ('axial','coronal','sagittal'):
            # self.path is available from kwargs of ItemList superclass
            fn  = self.path/tv/plane/(case + '.npy')
            res = self.open(fn)
            imagearrays.append(res)
        assert len(imagearrays) == 3
        return MRNetCase(*imagearrays)

    # since subclassing ItemList rather than ImageList, need an open method
    def open(self, fn): return np.load(fn)

    # TODO: reconstruct
    def reconstruct(self, t, x):
        # I think t is the tensor corresponding to the .data attribute of MRNetCase
        # and x is the tuple of tensors corresponding to the .obj attribute of MRNetCase
        # not sure if x is an MRNetCase or if x is MRNetCase.obj
        # the result of reconstruct should be to 
        # "return the same kind of object as .get returns"
        # which is a MRNetCase 
        # and to build that, the entire tuple of tensors x is required
        return MRNetCase(x) # or maybe MRNetCase(x.obj)

    # TODO: analyze_pred

    @classmethod
    def from_folder(cls, path:PathOrStr='.', extensions:Collection[str]=['.npy'], **kwargs)->'MRNetCaseList':
        "Get the filenames for all MRNet cases, assuming directory structure unchanged from MRNet data download"
        return super().from_folder(path=path, extensions=extensions, **kwargs)


    # advanced show methods
    # show_xys
    # show_xyzs

