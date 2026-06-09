train = False 


import torch

# Check if GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print(f"GPU is available. Using: {torch.cuda.get_device_name(0)}")
else:
    print("GPU not available. Using CPU.")


import os
import numpy as np
import nibabel as nib
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)

class Image(object):
    """
    Compact version of SCT's Image Class (https://github.com/spinalcordtoolbox/spinalcordtoolbox/blob/master/spinalcordtoolbox/image.py#L245)
    Create an object that behaves similarly to nibabel's image object. Useful additions include: dims, change_orientation and getNonZeroCoordinates.
    """

    def __init__(self, param=None, hdr=None, orientation=None, absolutepath=None, dim=None):
        """
        :param param: string indicating a path to a image file or an `Image` object.
        """

        # initialization of all parameters
        self.affine = None
        self.data = None
        self._path = None
        self.ext = ""

        if absolutepath is not None:
            self._path = os.path.abspath(absolutepath)
        
        # Case 1: load an image from file
        if isinstance(param, str):
            self.loadFromPath(param)
        # Case 2: create a copy of an existing `Image` object
        elif isinstance(param, type(self)):
            self.copy(param)
        # Case 3: create a blank image from a list of dimensions
        elif isinstance(param, list):
            self.data = np.zeros(param)
            self.hdr = hdr.copy() if hdr is not None else nib.Nifti1Header()
            self.hdr.set_data_shape(self.data.shape)
        # Case 4: create an image from an existing data array
        elif isinstance(param, (np.ndarray, np.generic)):
            self.data = param
            self.hdr = hdr.copy() if hdr is not None else nib.Nifti1Header()
            self.hdr.set_data_shape(self.data.shape)
        else:
            raise TypeError('Image constructor takes at least one argument.')
    
        # Fix any mismatch between the array's datatype and the header datatype
        self.fix_header_dtype()

    @property
    def dim(self):
        return get_dimension(self)
    
    @property
    def orientation(self):
        return get_orientation(self)
    
    @property
    def absolutepath(self):
        """
        Storage path (either actual or potential)

        Notes:

        - As several tools perform chdir() it's very important to have absolute paths
        - When set, if relative:

          - If it already existed, it becomes a new basename in the old dirname
          - Else, it becomes absolute (shortcut)

        Usually not directly touched (use `Image.save`), but in some cases it's
        the best way to set it.
        """
        return self._path
    
    @absolutepath.setter
    def absolutepath(self, value):
        if value is None:
            self._path = None
            return
        elif not os.path.isabs(value) and self._path is not None:
            value = os.path.join(os.path.dirname(self._path), value)
        elif not os.path.isabs(value):
            value = os.path.abspath(value)
        self._path = value
    
    @property
    def header(self):
        return self.hdr

    @header.setter
    def header(self, value):
        self.hdr = value

    def __deepcopy__(self, memo):
        return type(self)(deepcopy(self.data, memo), deepcopy(self.hdr, memo), deepcopy(self.orientation, memo), deepcopy(self.absolutepath, memo), deepcopy(self.dim, memo))

    def copy(self, image=None):
        if image is not None:
            self.affine = deepcopy(image.affine)
            self.data = deepcopy(image.data)
            self.hdr = deepcopy(image.hdr)
            self._path = deepcopy(image._path)
        else:
            return deepcopy(self)

    def loadFromPath(self, path):
        """
        This function load an image from an absolute path using nibabel library

        :param path: path of the file from which the image will be loaded
        :return:
        """

        self.absolutepath = os.path.abspath(path)
        im_file = nib.load(self.absolutepath, mmap=True)
        self.affine = im_file.affine.copy()
        self.data = np.asanyarray(im_file.dataobj)
        self.hdr = im_file.header.copy()
        if path != self.absolutepath:
            logger.debug("Loaded %s (%s) orientation %s shape %s", path, self.absolutepath, self.orientation, self.data.shape)
        else:
            logger.debug("Loaded %s orientation %s shape %s", path, self.orientation, self.data.shape)

    def change_orientation(self, orientation, inverse=False):
        """
        Change orientation on image (in-place).

        :param orientation: orientation string (SCT "from" convention)

        :param inverse: if you think backwards, use this to specify that you actually\
                        want to transform *from* the specified orientation, not *to*\
                        it.

        """
        change_orientation(self, orientation, self, inverse=inverse)
        return self
    
    def getNonZeroCoordinates(self, sorting=None, reverse_coord=False):
        """
        This function return all the non-zero coordinates that the image contains.
        Coordinate list can also be sorted by x, y, z, or the value with the parameter sorting='x', sorting='y', sorting='z' or sorting='value'
        If reverse_coord is True, coordinate are sorted from larger to smaller.

        Removed Coordinate object
        """
        n_dim = 1
        if self.dim[3] == 1:
            n_dim = 3
        else:
            n_dim = 4
        if self.dim[2] == 1:
            n_dim = 2

        if n_dim == 3:
            X, Y, Z = (self.data > 0).nonzero()
            list_coordinates = [[X[i], Y[i], Z[i], self.data[X[i], Y[i], Z[i]]] for i in range(0, len(X))]
        elif n_dim == 2:
            try:
                X, Y = (self.data > 0).nonzero()
                list_coordinates = [[X[i], Y[i], 0, self.data[X[i], Y[i]]] for i in range(0, len(X))]
            except ValueError:
                X, Y, Z = (self.data > 0).nonzero()
                list_coordinates = [[X[i], Y[i], 0, self.data[X[i], Y[i], 0]] for i in range(0, len(X))]

        if sorting is not None:
            if reverse_coord not in [True, False]:
                raise ValueError('reverse_coord parameter must be a boolean')

            if sorting == 'x':
                list_coordinates = sorted(list_coordinates, key=lambda el: el[0], reverse=reverse_coord)
            elif sorting == 'y':
                list_coordinates = sorted(list_coordinates, key=lambda el: el[1], reverse=reverse_coord)
            elif sorting == 'z':
                list_coordinates = sorted(list_coordinates, key=lambda el: el[2], reverse=reverse_coord)
            elif sorting == 'value':
                list_coordinates = sorted(list_coordinates, key=lambda el: el[3], reverse=reverse_coord)
            else:
                raise ValueError("sorting parameter must be either 'x', 'y', 'z' or 'value'")

        return list_coordinates
    
    def change_type(self, dtype):
        """
        Change data type on image.

        Note: the image path is voided.
        """
        change_type(self, dtype, self)
        return self
    
    def fix_header_dtype(self):
        """
        Change the header dtype to the match the datatype of the array.
        """
        # Using bool for nibabel headers is unsupported, so use uint8 instead:
        # `nibabel.spatialimages.HeaderDataError: data dtype "bool" not supported`
        dtype_data = self.data.dtype
        if dtype_data == bool:
            dtype_data = np.uint8

        dtype_header = self.hdr.get_data_dtype()
        if dtype_header != dtype_data:
            logger.warning(f"Image header specifies datatype '{dtype_header}', but array is of type "
                           f"'{dtype_data}'. Header metadata will be overwritten to use '{dtype_data}'.")
            self.hdr.set_data_dtype(dtype_data)
    
    def save(self, path=None, dtype=None, verbose=1, mutable=False):
        """
        Write an image in a nifti file

        :param path: Where to save the data, if None it will be taken from the\
                     absolutepath member.\
                     If path is a directory, will save to a file under this directory\
                     with the basename from the absolutepath member.

        :param dtype: if not set, the image is saved in the same type as input data\
                      if 'minimize', image storage space is minimized\
                        (2, 'uint8', np.uint8, "NIFTI_TYPE_UINT8"),\
                        (4, 'int16', np.int16, "NIFTI_TYPE_INT16"),\
                        (8, 'int32', np.int32, "NIFTI_TYPE_INT32"),\
                        (16, 'float32', np.float32, "NIFTI_TYPE_FLOAT32"),\
                        (32, 'complex64', np.complex64, "NIFTI_TYPE_COMPLEX64"),\
                        (64, 'float64', np.float64, "NIFTI_TYPE_FLOAT64"),\
                        (256, 'int8', np.int8, "NIFTI_TYPE_INT8"),\
                        (512, 'uint16', np.uint16, "NIFTI_TYPE_UINT16"),\
                        (768, 'uint32', np.uint32, "NIFTI_TYPE_UINT32"),\
                        (1024,'int64', np.int64, "NIFTI_TYPE_INT64"),\
                        (1280, 'uint64', np.uint64, "NIFTI_TYPE_UINT64"),\
                        (1536, 'float128', _float128t, "NIFTI_TYPE_FLOAT128"),\
                        (1792, 'complex128', np.complex128, "NIFTI_TYPE_COMPLEX128"),\
                        (2048, 'complex256', _complex256t, "NIFTI_TYPE_COMPLEX256"),

        :param mutable: whether to update members with newly created path or dtype
        """
        if mutable:  # do all modifications in-place
            # Case 1: `path` not specified
            if path is None:
                if self.absolutepath:  # Fallback to the original filepath
                    path = self.absolutepath
                else:
                    raise ValueError("Don't know where to save the image (no absolutepath or path parameter)")
            # Case 2: `path` points to an existing directory
            elif os.path.isdir(path):
                if self.absolutepath:  # Use the original filename, but save to the directory specified by `path`
                    path = os.path.join(os.path.abspath(path), os.path.basename(self.absolutepath))
                else:
                    raise ValueError("Don't know where to save the image (path parameter is dir, but absolutepath is "
                                     "missing)")
            # Case 3: `path` points to a file (or a *nonexistent* directory) so use its value as-is
            #    (We're okay with letting nonexistent directories slip through, because it's difficult to distinguish
            #     between nonexistent directories and nonexistent files. Plus, `nibabel` will catch any further errors.)
            else:
                pass

            if os.path.isfile(path) and verbose:
                logger.warning("File %s already exists. Will overwrite it.", path)
            if os.path.isabs(path):
                logger.debug("Saving image to %s orientation %s shape %s",
                             path, self.orientation, self.data.shape)
            else:
                logger.debug("Saving image to %s (%s) orientation %s shape %s",
                             path, os.path.abspath(path), self.orientation, self.data.shape)

            # Now that `path` has been set and log messages have been written, we can assign it to the image itself
            self.absolutepath = os.path.abspath(path)

            if dtype is not None:
                self.change_type(dtype)

            if self.hdr is not None:
                self.hdr.set_data_shape(self.data.shape)
                self.fix_header_dtype()

            # nb. that copy() is important because if it were a memory map, save() would corrupt it
            dataobj = self.data.copy()
            affine = None
            header = self.hdr.copy() if self.hdr is not None else None
            nib.save(nib.nifti1.Nifti1Image(dataobj, affine, header), self.absolutepath)
            if not os.path.isfile(self.absolutepath):
                raise RuntimeError(f"Couldn't save image to {self.absolutepath}")
        else:
            # if we're not operating in-place, then make any required modifications on a throw-away copy
            self.copy().save(path, dtype, verbose, mutable=True)
        return self


class SlicerOneAxis(object):
    """
    Image slicer to use when you don't care about the 2D slice orientation,
    and don't want to specify them.
    The slicer will just iterate through the right axis that corresponds to
    its specification.

    Can help getting ranges and slice indices.

    Copied from https://github.com/spinalcordtoolbox/spinalcordtoolbox/image.py
    """

    def __init__(self, im, axis="IS"):
        opposite_character = {'L': 'R', 'R': 'L', 'A': 'P', 'P': 'A', 'I': 'S', 'S': 'I'}
        axis_labels = "LRPAIS"
        if len(axis) != 2:
            raise ValueError()
        if axis[0] not in axis_labels:
            raise ValueError()
        if axis[1] not in axis_labels:
            raise ValueError()
        if axis[0] != opposite_character[axis[1]]:
            raise ValueError()

        for idx_axis in range(2):
            dim_nr = im.orientation.find(axis[idx_axis])
            if dim_nr != -1:
                break
        if dim_nr == -1:
            raise ValueError()

        # SCT convention
        from_dir = im.orientation[dim_nr]
        self.direction = +1 if axis[0] == from_dir else -1
        self.nb_slices = im.dim[dim_nr]
        self.im = im
        self.axis = axis
        self._slice = lambda idx: tuple([(idx if x in axis else slice(None)) for x in im.orientation])

    def __len__(self):
        return self.nb_slices

    def __getitem__(self, idx):
        """

        :return: an image slice, at slicing index idx
        :param idx: slicing index (according to the slicing direction)
        """
        if isinstance(idx, slice):
            raise NotImplementedError()

        if idx >= self.nb_slices:
            raise IndexError("I just have {} slices!".format(self.nb_slices))

        if self.direction == -1:
            idx = self.nb_slices - 1 - idx

        return self.im.data[self._slice(idx)]

def get_dimension(im_file, verbose=1):
    """
    Copied from https://github.com/spinalcordtoolbox/spinalcordtoolbox/

    Get dimension from Image or nibabel object. Manages 2D, 3D or 4D images.

    :param: im_file: Image or nibabel object
    :return: nx, ny, nz, nt, px, py, pz, pt
    """
    if not isinstance(im_file, (nib.nifti1.Nifti1Image, Image)):
        raise TypeError("The provided image file is neither a nibabel.nifti1.Nifti1Image instance nor an Image instance")
    # initializating ndims [nx, ny, nz, nt] and pdims [px, py, pz, pt]
    ndims = [1, 1, 1, 1]
    pdims = [1, 1, 1, 1]
    data_shape = im_file.header.get_data_shape()
    zooms = im_file.header.get_zooms()
    for i in range(min(len(data_shape), 4)):
        ndims[i] = data_shape[i]
        pdims[i] = zooms[i]
    return *ndims, *pdims


def change_orientation(im_src, orientation, im_dst=None, inverse=False):
    """
    Copied from https://github.com/spinalcordtoolbox/spinalcordtoolbox/

    :param im_src: source image
    :param orientation: orientation string (SCT "from" convention)
    :param im_dst: destination image (can be the source image for in-place
                   operation, can be unset to generate one)
    :param inverse: if you think backwards, use this to specify that you actually
                    want to transform *from* the specified orientation, not *to* it.
    :return: an image with changed orientation

    .. note::
        - the resulting image has no path member set
        - if the source image is < 3D, it is reshaped to 3D and the destination is 3D
    """

    if len(im_src.data.shape) < 3:
        pass  # Will reshape to 3D
    elif len(im_src.data.shape) == 3:
        pass  # OK, standard 3D volume
    elif len(im_src.data.shape) == 4:
        pass  # OK, standard 4D volume
    elif len(im_src.data.shape) == 5 and im_src.header.get_intent()[0] == "vector":
        pass  # OK, physical displacement field
    else:
        raise NotImplementedError("Don't know how to change orientation for this image")

    im_src_orientation = im_src.orientation
    im_dst_orientation = orientation
    if inverse:
        im_src_orientation, im_dst_orientation = im_dst_orientation, im_src_orientation

    perm, inversion = _get_permutations(im_src_orientation, im_dst_orientation)

    if im_dst is None:
        im_dst = im_src.copy()
        im_dst._path = None

    im_src_data = im_src.data
    if len(im_src_data.shape) < 3:
        im_src_data = im_src_data.reshape(tuple(list(im_src_data.shape) + ([1] * (3 - len(im_src_data.shape)))))

    # Update data by performing inversions and swaps

    # axes inversion (flip)
    data = im_src_data[::inversion[0], ::inversion[1], ::inversion[2]]

    # axes manipulations (transpose)
    if perm == [1, 0, 2]:
        data = np.swapaxes(data, 0, 1)
    elif perm == [2, 1, 0]:
        data = np.swapaxes(data, 0, 2)
    elif perm == [0, 2, 1]:
        data = np.swapaxes(data, 1, 2)
    elif perm == [2, 0, 1]:
        data = np.swapaxes(data, 0, 2)  # transform [2, 0, 1] to [1, 0, 2]
        data = np.swapaxes(data, 0, 1)  # transform [1, 0, 2] to [0, 1, 2]
    elif perm == [1, 2, 0]:
        data = np.swapaxes(data, 0, 2)  # transform [1, 2, 0] to [0, 2, 1]
        data = np.swapaxes(data, 1, 2)  # transform [0, 2, 1] to [0, 1, 2]
    elif perm == [0, 1, 2]:
        # do nothing
        pass
    else:
        raise NotImplementedError()

    # Update header

    im_src_aff = im_src.hdr.get_best_affine()
    aff = nib.orientations.inv_ornt_aff(
        np.array((perm, inversion)).T,
        im_src_data.shape)
    im_dst_aff = np.matmul(im_src_aff, aff)

    im_dst.header.set_qform(im_dst_aff)
    im_dst.header.set_sform(im_dst_aff)
    im_dst.header.set_data_shape(data.shape)
    im_dst.data = data

    return im_dst


def _get_permutations(im_src_orientation, im_dst_orientation):
    """
    Copied from https://github.com/spinalcordtoolbox/spinalcordtoolbox/

    :param im_src_orientation str: Orientation of source image. Example: 'RPI'
    :param im_dest_orientation str: Orientation of destination image. Example: 'SAL'
    :return: list of axes permutations and list of inversions to achieve an orientation change
    """

    opposite_character = {'L': 'R', 'R': 'L', 'A': 'P', 'P': 'A', 'I': 'S', 'S': 'I'}

    perm = [0, 1, 2]
    inversion = [1, 1, 1]
    for i, character in enumerate(im_src_orientation):
        try:
            perm[i] = im_dst_orientation.index(character)
        except ValueError:
            perm[i] = im_dst_orientation.index(opposite_character[character])
            inversion[i] = -1

    return perm, inversion


def get_orientation(im):
    """
    Copied from https://github.com/spinalcordtoolbox/spinalcordtoolbox/

    :param im: an Image
    :return: reference space string (ie. what's in Image.orientation)
    """
    res = "".join(nib.orientations.aff2axcodes(im.hdr.get_best_affine()))
    return orientation_string_nib2sct(res)


def orientation_string_nib2sct(s):
    """
    Copied from https://github.com/spinalcordtoolbox/spinalcordtoolbox/

    :return: SCT reference space code from nibabel one
    """
    opposite_character = {'L': 'R', 'R': 'L', 'A': 'P', 'P': 'A', 'I': 'S', 'S': 'I'}
    return "".join([opposite_character[x] for x in s])


def change_type(im_src, dtype, im_dst=None):
    """
    Change the voxel type of the image

    :param dtype:    if not set, the image is saved in standard type\
                    if 'minimize', image space is minimize\
                    if 'minimize_int', image space is minimize and values are approximated to integers\
                    (2, 'uint8', np.uint8, "NIFTI_TYPE_UINT8"),\
                    (4, 'int16', np.int16, "NIFTI_TYPE_INT16"),\
                    (8, 'int32', np.int32, "NIFTI_TYPE_INT32"),\
                    (16, 'float32', np.float32, "NIFTI_TYPE_FLOAT32"),\
                    (32, 'complex64', np.complex64, "NIFTI_TYPE_COMPLEX64"),\
                    (64, 'float64', np.float64, "NIFTI_TYPE_FLOAT64"),\
                    (256, 'int8', np.int8, "NIFTI_TYPE_INT8"),\
                    (512, 'uint16', np.uint16, "NIFTI_TYPE_UINT16"),\
                    (768, 'uint32', np.uint32, "NIFTI_TYPE_UINT32"),\
                    (1024,'int64', np.int64, "NIFTI_TYPE_INT64"),\
                    (1280, 'uint64', np.uint64, "NIFTI_TYPE_UINT64"),\
                    (1536, 'float128', _float128t, "NIFTI_TYPE_FLOAT128"),\
                    (1792, 'complex128', np.complex128, "NIFTI_TYPE_COMPLEX128"),\
                    (2048, 'complex256', _complex256t, "NIFTI_TYPE_COMPLEX256"),
    :return:

    Copied from https://github.com/spinalcordtoolbox/spinalcordtoolbox/
    """

    if im_dst is None:
        im_dst = im_src.copy()
        im_dst._path = None

    if dtype is None:
        return im_dst

    # get min/max from input image
    min_in = np.nanmin(im_src.data)
    max_in = np.nanmax(im_src.data)

    # find optimum type for the input image
    if dtype in ('minimize', 'minimize_int'):
        # warning: does not take intensity resolution into account, neither complex voxels

        # check if voxel values are real or integer
        isInteger = True
        if dtype == 'minimize':
            for vox in im_src.data.flatten():
                if int(vox) != vox:
                    isInteger = False
                    break

        if isInteger:
            if min_in >= 0:  # unsigned
                if max_in <= np.iinfo(np.uint8).max:
                    dtype = np.uint8
                elif max_in <= np.iinfo(np.uint16):
                    dtype = np.uint16
                elif max_in <= np.iinfo(np.uint32).max:
                    dtype = np.uint32
                elif max_in <= np.iinfo(np.uint64).max:
                    dtype = np.uint64
                else:
                    raise ValueError("Maximum value of the image is to big to be represented.")
            else:
                if max_in <= np.iinfo(np.int8).max and min_in >= np.iinfo(np.int8).min:
                    dtype = np.int8
                elif max_in <= np.iinfo(np.int16).max and min_in >= np.iinfo(np.int16).min:
                    dtype = np.int16
                elif max_in <= np.iinfo(np.int32).max and min_in >= np.iinfo(np.int32).min:
                    dtype = np.int32
                elif max_in <= np.iinfo(np.int64).max and min_in >= np.iinfo(np.int64).min:
                    dtype = np.int64
                else:
                    raise ValueError("Maximum value of the image is to big to be represented.")
        else:
            # if max_in <= np.finfo(np.float16).max and min_in >= np.finfo(np.float16).min:
            #    type = 'np.float16' # not supported by nibabel
            if max_in <= np.finfo(np.float32).max and min_in >= np.finfo(np.float32).min:
                dtype = np.float32
            elif max_in <= np.finfo(np.float64).max and min_in >= np.finfo(np.float64).min:
                dtype = np.float64

        dtype = to_dtype(dtype)
    else:
        dtype = to_dtype(dtype)

        # if output type is int, check if it needs intensity rescaling
        if "int" in dtype.name:
            # get min/max from output type
            min_out = np.iinfo(dtype).min
            max_out = np.iinfo(dtype).max
            # before rescaling, check if there would be an intensity overflow

            if (min_in < min_out) or (max_in > max_out):
                # This condition is important for binary images since we do not want to scale them
                logger.warning(f"To avoid intensity overflow due to convertion to +{dtype.name}+, intensity will be rescaled to the maximum quantization scale")
                # rescale intensity
                data_rescaled = im_src.data * (max_out - min_out) / (max_in - min_in)
                im_dst.data = data_rescaled - (data_rescaled.min() - min_out)

    # change type of data in both numpy array and nifti header
    im_dst.data = getattr(np, dtype.name)(im_dst.data)
    im_dst.hdr.set_data_dtype(dtype)
    return im_dst


def to_dtype(dtype):
    """
    Take a dtypeification and return an np.dtype

    :param dtype: dtypeification (string or np.dtype or None are supported for now)
    :return: dtype or None

    Copied from https://github.com/spinalcordtoolbox/spinalcordtoolbox/
    """
    # TODO add more or filter on things supported by nibabel

    if dtype is None:
        return None
    if isinstance(dtype, type):
        if isinstance(dtype(0).dtype, np.dtype):
            return dtype(0).dtype
    if isinstance(dtype, np.dtype):
        return dtype
    if isinstance(dtype, str):
        return np.dtype(dtype)

    raise TypeError("data type {}: {} not understood".format(dtype.__class__, dtype))


def zeros_like(img, dtype=None):
    """

    :param img: reference image
    :param dtype: desired data type (optional)
    :return: an Image with the same shape and header, filled with zeros

    Similar to numpy.zeros_like(), the goal of the function is to show the developer's
    intent and avoid doing a copy, which is slower than initialization with a constant.

    Copied from https://github.com/spinalcordtoolbox/spinalcordtoolbox/image.py
    """
    zimg = Image(np.zeros_like(img.data), hdr=img.hdr.copy())
    if dtype is not None:
        zimg.change_type(dtype)
    return zimg


def empty_like(img, dtype=None):
    """
    :param img: reference image
    :param dtype: desired data type (optional)
    :return: an Image with the same shape and header, whose data is uninitialized

    Similar to numpy.empty_like(), the goal of the function is to show the developer's
    intent and avoid touching the allocated memory, because it will be written to
    afterwards.

    Copied from https://github.com/spinalcordtoolbox/spinalcordtoolbox/image.py
    """
    dst = change_type(img, dtype)
    return dst


def find_zmin_zmax(im, threshold=0.1):
    """
    Find the min (and max) z-slice index below which (and above which) slices only have voxels below a given threshold.

    :param im: Image object
    :param threshold: threshold to apply before looking for zmin/zmax, typically corresponding to noise level.
    :return: [zmin, zmax]

    Copied from https://github.com/spinalcordtoolbox/spinalcordtoolbox/image.py
    """
    slicer = SlicerOneAxis(im, axis="IS")

    # Make sure image is not empty
    if not np.any(slicer):
        logger.error('Input image is empty')

    # Iterate from bottom to top until we find data
    for zmin in range(0, len(slicer)):
        if np.any(slicer[zmin] > threshold):
            break

    # Conversely from top to bottom
    for zmax in range(len(slicer) - 1, zmin, -1):
        if np.any(slicer[zmax] > threshold):
            break

    return zmin, zmax


# Installation of  libraries
!pip install --no-deps /kaggle/input/kaggle-wheels/gryds-0.0.9-py3-none-any.whl
!pip install --no-deps /kaggle/input/kaggle-wheels/monai-1.3.2-py3-none-any.whl
!pip install --no-deps /kaggle/input/kaggle-wheels/torchio-0.19.9-py2.py3-none-any.whl
!pip install --no-deps /kaggle/input/cc3d-wheel/connected_components_3d-3.18.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
!pip install --no-deps /kaggle/input/kaggle-wheels/totalspineseg-20241005-py3-none-any.whl
#!pip install --no-deps /kaggle/input/kaggle-wheels/gryds-0.0.9-py3-none-any.whl
#!pip install --no-deps /kaggle/input/kaggle-wheels/monai-1.3.2-py3-none-any.whl
#!pip install --no-deps /kaggle/input/kaggle-wheels/torchio-0.19.9-py2.py3-none-any.whl
!pip install --no-deps /kaggle/input/kaggle-wheels/nnunetv2-2.5.1-py3-none-any.whl
!pip install --no-deps /kaggle/input/acvl-utils/acvl_utils-0.2-py3-none-any.whl
!pip install --no-deps /kaggle/input/batchgenerators/batchgenerators-0.25-py3-none-any.whl
!pip install --no-deps /kaggle/input/nnunet-wheels/batchgeneratorsv2-0.2.1-py3-none-any.whl
!pip install --no-deps /kaggle/input/nnunet-wheels/fft_conv_pytorch-1.2.0-py3-none-any.whl
!pip install --no-deps /kaggle/input/dynamic-net/dynamic_network_architectures-0.3.1-py3-none-any.whl
#!pip install --no-deps /kaggle/input/cc3d-wheel/connected_components_3d-3.18.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
!pip install --no-deps /kaggle/input/pillow-wheel/pillow-10.4.0-cp310-cp310-manylinux_2_28_x86_64.whl


! totalspineseg -h


os.makedirs("TotalSpineSeg", exist_ok = True)
os.system("cp -r /kaggle/input/m/simonqueric/totalspineseg/other/default/6/totalspineseg/totalspineseg TotalSpineSeg")
os.system("python3 -m venv /kaggle/TotalSpineSeg/venv")
os.system('bash -c "source /kaggle/TotalSpineSeg/venv/bin/activate"')
os.makedirs('/kaggle/working/TotalSpineSeg/tss_input', exist_ok=True)
os.system('export TOTALSPINESEG="$(realpath totalspineseg)"')
os.system('export TOTALSPINESEG_DATA="$(realpath data)"')


def get_subjects(source_dir, batch_size=50):
    """Get a list of batches of subjects filtered by the filter_func."""
    all_subjects = [sub for sub in os.listdir(source_dir)]
    return all_subjects


def run_totalspineseg(source_dir):
    """Applies TotalSpineSeg to every scan in the source_dir and saves the segmentations."""
    # Define temporary directories
    tss_temp_dir = "TotalSpineSeg/tss_input"
    output_temp = "temp_output_data"
    failed_subjects = []

    # Get all batches of subjects
    subjects = get_subjects(source_dir, batch_size=50)
   
    # Process each batch
    
    os.makedirs(tss_temp_dir, exist_ok=True)
    os.makedirs(output_temp, exist_ok=True)

    for subdir in subjects:
        try:
            anat_path = os.path.join(source_dir, subdir, 'anat')
            if os.path.exists(anat_path):
                for file in os.listdir(anat_path):
                    file_path = os.path.join(anat_path, file)
                    if os.path.isfile(file_path) and 'ax' not in file_path and 'total_seg' not in file_path and 'T2' not in file_path:
                        shutil.copy(file_path, tss_temp_dir)
                        print('File copied successfully.')
        except Exception as e:
            print(f"Failed processing subject {subdir}: {e}")
            failed_subjects.append(subdir)

    # Run TotalSpineSeg segmentation
    os.system(f"totalspineseg --data-dir /kaggle/working/{tss_temp_dir} /kaggle/working/{tss_temp_dir} /kaggle/working/{output_temp} --step1")
    

    # Move segmentations back into original data structure
    segmentations_into_anat(output_temp, source_dir)

    # Clean up temporary directories
    os.system("rm /kaggle/working/TotalSpineSeg/tss_input/*.nii.gz")
    shutil.rmtree(output_temp)


def segmentations_into_anat(output_folder, nii_folder):
    """Send the segmentations into the folder with the nii volumes."""
    seg_folder = os.path.join(output_folder, "step1_output")
    segmentations = os.listdir(seg_folder)

    for segmentation in segmentations:
        id_patient = segmentation.split('_')[0]
        patient_folder = os.path.join(nii_folder, id_patient, 'anat')

        if os.path.exists(patient_folder):
            source_path = os.path.join(seg_folder, segmentation)
            modified_segmentation = segmentation.replace('.nii.gz', '_total_seg.nii.gz')
            destination_path = os.path.join(patient_folder, modified_segmentation)
            shutil.copy(source_path, destination_path)






import os
import sys
import shutil
import csv
import subprocess
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
import torch.optim as optim
from torch.nn import CrossEntropyLoss

from tqdm import tqdm
import monai
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, ConcatItemsd,
    ToTensord, RandRotate90d, RandFlipd, SpatialPadd, CenterSpatialCropd,
    NormalizeIntensityd, RandScaleIntensityd, RandShiftIntensityd, RandRotated,
    Spacingd, RandSpatialCropd, RandBiasFieldd, Flipd, SpatialCropd, Transform, 
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, ConcatItemsd,
    ToTensord, SpatialPadd, CenterSpatialCropd, NormalizeIntensityd,
    RandRotated, RandSpatialCropd, RandBiasFieldd, Lambdad, Transform,
    RandGaussianNoised, RandAffined, RandZoomd, Rand3DElasticd, Spacingd
)
from monai.networks.nets import DenseNet201, ResNet
from monai.data import Dataset

import seaborn as sns
import matplotlib.pyplot as plt
import nibabel as nib
import torchio as tio
from sklearn.metrics import confusion_matrix
import argparse
from scipy.ndimage import center_of_mass
from skimage.measure import regionprops


import shutil 
import os 

shutil.copytree('/kaggle/input/dcm2nii', '/kaggle/working/dcm2nii', dirs_exist_ok=True)


! chmod +x /kaggle/working/dcm2nii/dcm2niix/build/bin/dcm2niix
os.environ['PATH'] += ':/kaggle/working/dcm2nii/dcm2niix/build/bin/'


# use a subprocess to convert the dicom images to nifti format, requires the output path
def convert_dicom_to_nifti(subject_id, series_uid, input_path, output_path):
    """
    Convert DICOM images to NIfTI format using dcm2niix.
    
    Parameters:
    subject_id (str): The subject identifier.
    series_uid (str): The series instance UID.
    input_path (str): Path to the DICOM images directory.
    output_path (str): Path to the output directory for NIfTI files.
    """
    input_file = os.path.join(input_path, subject_id, series_uid)
    output_file = os.path.join(output_path, f"{subject_id}-{series_uid}")
    if not os.path.exists(output_file):
        os.makedirs(output_file)

    dcm2niix_command = f"dcm2niix -z y -m 2 -o {output_file} {input_file}"
    
    try:
        subprocess.run(dcm2niix_command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error converting {input_file}: {e}")
        return None

# do not apply this function to the axial acquisitions, as it will merge different acquisitions with different orientations
def merge_nifti_volumes(output_path, subject_id, series_uid):
    """
    Merge NIfTI volumes in the Z direction and save the merged volume if more than one volume exists.
    Otherwise, save the single volume directly.
    Rename the merged volume to the specified format.
    
    Parameters:
    output_filename (str): Output filename for the merged NIfTI volume.
    output_path (str): Path to the output directory for merged NIfTI volume.
    subject_id (str): The subject identifier.
    series_uid (str): The series instance UID.
    """
    output_path_for_merge = os.path.join(output_path, f"{subject_id}-{series_uid}")

    filenames = glob.glob(os.path.join(output_path_for_merge, '*.nii.gz'))
    filenames.sort()
    new_paths = []
    if len(filenames) > 1:
        for filename in filenames: 
            merged_filename = f"sub-{subject_id}_run-{series_uid}_{filename[-14:]}"
            merged_path = os.path.join(output_path, merged_filename)  # Changed to output_folder
            os.rename(filename, merged_path)
            new_paths.append(merged_path)
    elif len(filenames) == 1:
        merged_filename = f"sub-{subject_id}_run-{series_uid}.nii.gz"
        merged_path = os.path.join(output_path, merged_filename)  # Changed to output_folder
        os.rename(filenames[0], merged_path)
        new_paths.append(merged_path)
    return new_paths


# reorient the image to a common orientation "LPI"
def reorient(image):

    # Get image dtype from the image data (preferred over header dtype to avoid data loss)
    image_data_dtype = getattr(np, np.asanyarray(image.dataobj).dtype.name)

    # Rescale the image to the output dtype range if necessary
    # Modified from https://github.com/spinalcordtoolbox/spinalcordtoolbox/blob/6.3/spinalcordtoolbox/image.py#L1217
    if "int" in np.dtype(image_data_dtype).name:
        image_data = np.asanyarray(image.dataobj).astype(np.float64)
        image_min, image_max = image_data.min(), image_data.max()
        dtype_min, dtype_max = np.iinfo(image_data_dtype).min, np.iinfo(image_data_dtype).max
        if (image_min < dtype_min) or (dtype_max < image_max):
            data_rescaled = image_data * (dtype_max - dtype_min) / (image_max - image_min)
            image_data = data_rescaled - (data_rescaled.min() - dtype_min)
            image = nib.Nifti1Image(image_data.astype(image_data_dtype), image.affine, image.header)

    # Transform the image to the closest canonical orientation
    output_image = nib.as_closest_canonical(image)

    # Ensure correct image dtype, affine and header
    output_image = nib.Nifti1Image(
        np.asanyarray(output_image.dataobj).astype(image_data_dtype),
        output_image.affine, output_image.header
    )
    output_image.set_data_dtype(image_data_dtype)
    output_image.set_qform(output_image.affine)
    output_image.set_sform(output_image.affine)

    return output_image





def process_subject(subject_id, input_path, output_path, train, meta_obj):
    """
    Process DICOM to NIfTI conversion, merge volumes, and save with corrected orientation if applicable.
    
    Parameters:
    subject_id (str): The subject identifier.
    input_path (str): Path to the DICOM images directory.
    output_path (str): Path to the output directory for NIfTI files.
    train (DataFrame): DataFrame containing series information.
    meta_obj (dict): Metadata object with series information.
    """
    if subject_id not in train['study_id'].astype(str).values:
        return

    filtered_series = train[train['study_id'] == int(subject_id)].iloc[0]
    ptobj = meta_obj[str(filtered_series['study_id'])]

    if ptobj is None:
        return

    # create output directories if not existing
    os.makedirs(os.path.join(output_path, f'sub-{subject_id}'), exist_ok=True)
    os.makedirs(os.path.join(output_path, f'sub-{subject_id}', 'anat'), exist_ok=True)

    # process through each acquisition of the subject    
    for idx, series_uid in enumerate(ptobj['SeriesInstanceUIDs']):
        description = ptobj['SeriesDescriptions'][idx]

        convert_dicom_to_nifti(subject_id, series_uid, input_path, output_path)
        new_paths = merge_nifti_volumes(output_path, subject_id, series_uid)
        if 'Axial' in description and 'T2' in description:
                modality = 'T2w'
                acq = 'ax'
        elif 'Sagittal' in description and 'T1' in description:
            modality = 'T1w'
            acq = 'sag'
        elif 'Sagittal' in description and 'T2' in description:
            modality = 'T2w'
            acq = 'sag'
        else:
            continue

        corrected_nifti_path = os.path.join(output_path, f"sub-{subject_id}/anat/sub-{subject_id}_acq-{acq}_rec{series_uid}_{modality}")
        if len(new_paths) > 1 : 
            for merged_nifti_path in new_paths : 
                anat_img = nib.load(merged_nifti_path)
                anat_data = anat_img.get_fdata()
                anat_affine = anat_img.affine
                anat_header = anat_img.header

                new_affine = np.copy(anat_affine)
                anat_header.set_qform(new_affine, code=1)
                anat_header.set_sform(new_affine, code=1)

                base, ext = os.path.splitext(merged_nifti_path)

                new_path = corrected_nifti_path + base[-11:] + ext

                # reorient the image
                image = nib.Nifti1Image(anat_data, new_affine, header=anat_header)
                
                oriented_image = reorient(image)
                # then apply the resampling to the median values resolution for axial T2w images
                if acq == 'ax': 
                    """final_image = resample_nifti(oriented_image, target_spacing=(0.4, 0.4, 4.4), mode='linear')  
                    
                    nib.save(final_image, new_path)"""
                    nib.save(oriented_image, new_path)
                else:
                    """final_image = resample_nifti(oriented_image, target_spacing=(4.0, 0.4, 0.4), mode='linear')  

                    nib.save(final_image, new_path)"""
                    nib.save(oriented_image, new_path)

        else : 
            for merged_nifti_path in new_paths : 
                anat_img = nib.load(merged_nifti_path)
                anat_data = anat_img.get_fdata()
                anat_affine = anat_img.affine
                anat_header = anat_img.header

                new_affine = np.copy(anat_affine)
                anat_header.set_qform(new_affine, code=1)
                anat_header.set_sform(new_affine, code=1)
                new_path = corrected_nifti_path + '.nii.gz'
                # reorient the image
                image = nib.Nifti1Image(anat_data, new_affine, header=anat_header)
                oriented_image = reorient(image)

                # then apply the resampling to the median values resolution for axial T2w images
                if acq == 'ax': 
                    
                                
                    """final_image = resample_nifti(oriented_image, target_spacing=(0.4, 0.4, 4.4), mode='linear') 
                    nib.save(final_image, new_path)"""
                    nib.save(oriented_image, new_path)
                else:

                    """final_image = resample_nifti(oriented_image, target_spacing=(4.0, 0.4, 0.4), mode='linear')  
                    nib.save(final_image, new_path)"""
                    nib.save(oriented_image, new_path)
    

# Main function to run the processing
def BIDSification(input_folder, output_folder, csv_description):
    
    os.makedirs(output_folder, exist_ok=True)

    ### Create the dictionary based on the CSV file ###
    df_meta_f = pd.read_csv(csv_description, sep=',')
    subject_ids = np.unique(df_meta_f["study_id"].values)

    # List out all of the Studies we have on patients.
    part_1 = os.listdir(input_folder)
    part_1 = list(filter(lambda x: x.find('.DS') == -1, part_1))

    p1 = [(x, f"{input_folder}/{x}") for x in part_1]
    meta_obj = { p[0]: { 'folder_path': p[1], 
                        'SeriesInstanceUIDs': [] 
                    } 
                for p in p1 }

    for m in meta_obj:
        meta_obj[m]['SeriesInstanceUIDs'] = list( 
            filter(lambda x: x.find('.DS') == -1, 
                os.listdir(meta_obj[m]['folder_path'])
                )
        )
    # Grabs the corresponding series descriptions
    for k in tqdm(meta_obj):
        for s in meta_obj[k]['SeriesInstanceUIDs']:
            if 'SeriesDescriptions' not in meta_obj[k]:
                meta_obj[k]['SeriesDescriptions'] = []
            try:
                meta_obj[k]['SeriesDescriptions'].append(
                    df_meta_f[(df_meta_f['study_id'] == int(k)) & 
                    (df_meta_f['series_id'] == int(s))]['series_description'].iloc[0])
            except:
                None

    # Process subjects and set up directories
    
    for subject_id in tqdm(subject_ids):  # Adjust range as needed: 1975 subjects
        try: 
            subject_id = str(subject_id)
            
            # Create specific directories
            os.makedirs(os.path.join(output_folder, f'sub-{subject_id}'), exist_ok=True)
            os.makedirs(os.path.join(output_folder, f'sub-{subject_id}', 'anat'), exist_ok=True)

            # Process subject and set up directories
            process_subject(subject_id, input_folder, output_folder, df_meta_f, meta_obj)
        except: 
            print(f'failed preprocessing for {subject_id}')
 

    for item in os.listdir(output_folder):
        item_path = os.path.join(output_folder, item)
        
        # Check if the item is a directory and starts with "sub"
        if os.path.isdir(item_path) and item.startswith("sub"):
            continue  # Skip deletion for folders starting with "sub"
        
        # Delete the item (file or directory)
        if os.path.isfile(item_path):
            os.remove(item_path)
            print(f"Deleted file: {item_path}")
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)
            print(f"Deleted folder: {item_path}")




def get_shifted_point_along_disk(disk_mask):
    """
    Calcule un point décalé selon l'axe du disque en LPI.
    
    Args:
        disk_mask: Masque binaire 3D du disque
    
    Returns:
        point: numpy array des coordonnées (x,y,z) du point décalé
        disk_radius: rayon calculé du disque selon son axe
        direction_vector: vecteur normalisé indiquant la direction du disque
    """
    # Trouver le centre du disque
    centroid = center_of_mass(disk_mask)
    # Trouver la slice sagittale contenant le centre du disque
    sagittal_slice_idx = int(centroid[0])
    sagittal_slice = disk_mask[sagittal_slice_idx, :, :]
    
    # Calculer l'orientation sur la slice 2D
    props = regionprops(sagittal_slice.astype(int))[0]
    orientation = props.orientation  # en radians

    direction_vector = np.array([
        0,  # x reste inchangé
        -np.cos(orientation),   # y
        -np.sin(orientation)    # z
    ])
    
    # Normaliser le vecteur
    direction_vector = direction_vector / np.linalg.norm(direction_vector)
    
    # Calculer le rayon du disque (projection sur le vecteur)
    mask_points = np.array(np.where(disk_mask)).T
    centered_points = mask_points - centroid
    projections = np.abs(centered_points @ direction_vector)
    disk_radius = np.max(projections)
    
    # Calculer le point décalé
    shifted_point = centroid + direction_vector * disk_radius

    return shifted_point


## STEP 3 ##

# this is last part of the preprocessing pipeline
# its goal is to extract the patches from the nii volumes based on the segmentation 

def process_directory_other(main_dir):
    '''
    Transform the segmentations in main_dir folder to the image space to have the same origin, spacing, direction and shape as the image.

    Parameters
    main_dir: where to fetch the segmentations
    
    '''

    main_dir_path = Path(main_dir)
    
    # Iterate through each subdirectory (for each patient)
    for dirpath, dirnames, filenames in os.walk(main_dir_path):
        # Vérifier que nous sommes dans un dossier patient et qu'il y a un sous-dossier anat
       
        if "anat" in dirnames:
            
            anat_path = os.path.join(dirpath, "anat")
            
            # Obtenir la liste des fichiers dans le sous-dossier anat
            anat_filenames = os.listdir(anat_path)
            
        
            # Find sagittal T2w image
            sag_files = [f for f in anat_filenames if "acq-sag" in f and "T1w_total_seg" in f]
            if len(sag_files) == 0:
                
                continue
            
            sag_file = os.path.join(anat_path, sag_files[0])
            
            # Find and process all axial images
            for ax_file in anat_filenames:
                try: 
                    if "acq-ax" in ax_file and not "seg" in ax_file and not "patch" in ax_file:
                        ax_file_path = os.path.join(anat_path, ax_file)
                        
                        output_file_path = ax_file_path.replace(".nii.gz", "_total_seg.nii.gz")
                        
                        
                        # Call the transformation function
                        _transform_seg2image(ax_file_path, sag_file, output_file_path)
                except: 
                    print (ax_file)

def _transform_seg2image(
        image_path,
        seg_path,
        output_seg_path,
        override=False,
    ):
    '''
    Wrapper function to handle IO.
    '''
    image_path = Path(image_path)
    seg_path = Path(seg_path)
    output_seg_path = Path(output_seg_path)

    # If the output image already exists and we are not overriding it, return
    if not override and output_seg_path.exists():
        return

    # Check if the segmentation file exists
    if not seg_path.is_file():
        output_seg_path.is_file() and output_seg_path.unlink()
        return

    image = nib.load(image_path)
    seg = nib.load(seg_path)

    output_seg = transform_seg2image(image, seg)

    # Ensure correct segmentation dtype, affine and header
    output_seg = nib.Nifti1Image(
        np.asanyarray(output_seg.dataobj).round().astype(np.uint8),
        output_seg.affine, output_seg.header
    )
    output_seg.set_data_dtype(np.uint8)
    output_seg.set_qform(output_seg.affine)
    output_seg.set_sform(output_seg.affine)

    # Make sure output directory exists and save the segmentation
    output_seg_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(output_seg, output_seg_path)




def transform_seg2image(
        image,
        seg,
    ):
    '''
    Transform the segmentation to the image space to have the same origin, spacing, direction and shape as the image.

    Parameters
    ----------
    image : nibabel.Nifti1Image
        Image.
    seg : nibabel.Nifti1Image
        Segmentation.

    Returns
    -------
    nibabel.Nifti1Image
        Output segmentation.
    '''
    image_data = np.asanyarray(image.dataobj).astype(np.float64)
    seg_data = np.asanyarray(seg.dataobj).round().astype(np.uint8)

    # Make TorchIO images
    tio_img=tio.ScalarImage(tensor=image_data[None, ...], affine=image.affine)
    tio_seg=tio.LabelMap(tensor=seg_data[None, ...], affine=seg.affine)

    # Resample the segmentation to the image space
    tio_output_seg = tio.Resample(tio_img)(tio_seg)
    output_seg_data = tio_output_seg.data.numpy()[0, ...].astype(np.uint8)

    output_seg = nib.Nifti1Image(output_seg_data, image.affine, seg.header)

    return output_seg




# function for sagittal patches
def patch_extraction_foraminal(vol, mask, affine):
    """
    Extract two 3D patches from an MRI volume centered around mask's centroid
    
    Parameters:
    - vol: 3D numpy array representing the volume
    - mask: 3D segmentation mask 
    - affine: Affine matrix from the NIfTI file
    
    Returns:
    - patch1, patch2: Two 3D numpy array patches
    """
    i = 0

    D, H, W = vol.shape
    # mask = torch.Tensor(mask)
    # nonzero_indices = torch.nonzero(mask)
    
    # Calculate centroid of the mask and shift it along the disk axis
    centroid = get_shifted_point_along_disk(mask).astype(int)

    # Get voxel sizes from the affine matrix
    voxel_sizes = np.abs(np.diag(affine)[:3])
    
    patch_size_mm = {
        'd': 50,  # depth
        'h': 50,  # height
        'w': 50   # width
    }

    # Patch sizes (in voxels)
    patch_sizes_voxels = {
        'd': (patch_size_mm['d'] / voxel_sizes[0]).astype(int),
        'h': (patch_size_mm['h'] / voxel_sizes[1]).astype(int),
        'w': (patch_size_mm['w'] / voxel_sizes[2]).astype(int)
    }

    
    # Extract patches centered on centroid with posterior displacement
    patch1 = vol[
        max(0, centroid[0] + 1):min(D, centroid[0] + patch_sizes_voxels['d']//2 +1),
        max(0, centroid[1] - patch_sizes_voxels['h']//2):min(H, centroid[1] + patch_sizes_voxels['h']//2),
        max(0, centroid[2] - patch_sizes_voxels['w']//2):min(W, centroid[2] + patch_sizes_voxels['w']//2)
    ]
    
    patch2 = vol[
        max(0, centroid[0] -1 -patch_sizes_voxels['d']//2):min(D, centroid[0]-1),
        max(0, centroid[1] - patch_sizes_voxels['h']//2):min(H, centroid[1] + patch_sizes_voxels['h']//2),
        max(0, centroid[2] - patch_sizes_voxels['w']//2):min(W, centroid[2] + patch_sizes_voxels['w']//2)
    ]
    
    return patch1, patch2

# uses lists of sagittal images and segmentations to extract patches for each disc
def extract_and_save_sagittal_patches(sagittal_images, sagittal_segmentations, nii_folder, output_folder):
    # Match each axial image to its corresponding sagittal segmentation
    for img_name, seg_sag_name in zip(sagittal_images, sagittal_segmentations):
        if "patch" not in img_name:
            img_path = os.path.join(nii_folder, img_name)
            seg_sag_path = os.path.join(nii_folder, seg_sag_name)
            affine_ex = nib.load(img_path).affine
            
            # Load the volumetric image and sagittal segmentation
            vol = nib.load(img_path).get_fdata()
            seg_sag = nib.load(seg_sag_path).get_fdata()

            # Détection des disques dans la segmentation sagittale
            #The values to check are based on the classes in totalspineseg 
            disc_l5 = np.isin(seg_sag, [100]).astype(int)
            disc_l4 = np.isin(seg_sag, [95]).astype(int)
            disc_l3 = np.isin(seg_sag, [94]).astype(int)
            disc_l2 = np.isin(seg_sag, [93]).astype(int)
            disc_l1 = np.isin(seg_sag, [92]).astype(int)

            
            discs_dict = {
                "L1_L2": disc_l1,
                "L2_L3": disc_l2,
                "L3_L4": disc_l3,
                "L4_L5": disc_l4,
                "L5_S1": disc_l5
            }    

            # Extract and save patches for each disc
            for disc_name, disc_mask in discs_dict.items():
                if np.any(disc_mask):  # If the disc is found in the segmentation
                    # Extract the patch using the segmentation mask
                    
                    patch_img_left, patch_img_right = patch_extraction_foraminal(vol, disc_mask, affine_ex)
                    
                    if patch_img_left is not None or patch_img_right is not None:  # Proceed only if patch extraction was successful

                        # Construct the filename and file path
                        patch_img_filename_left = f"{img_name[:-7]}_{disc_name}_foramen_left_patch.nii.gz"
                        patch_img_filepath_left = os.path.join(output_folder, patch_img_filename_left)

                        patch_img_filename_right = f"{img_name[:-7]}_{disc_name}_foramen_right_patch.nii.gz"
                        patch_img_filepath_right = os.path.join(output_folder, patch_img_filename_right)

                        # Use the affine from the original volume to create the patch NIfTI image
                        original_affine = nib.load(img_path).affine
                        original_header = nib.load(img_path).header.copy()
                        patch_nifti_img_left = nib.Nifti1Image(patch_img_left, affine=original_affine)
                        patch_nifti_img_right = nib.Nifti1Image(patch_img_right, affine=original_affine)


                        q_code = int(original_header['qform_code'])
                        s_code = int(original_header['sform_code'])

                        patch_nifti_img_left.header.set_qform(original_affine, code=q_code)
                        patch_nifti_img_left.header.set_sform(original_affine, code=s_code)
                        patch_nifti_img_right.header.set_qform(original_affine, code=q_code)
                        patch_nifti_img_right.header.set_sform(original_affine, code=s_code)

                        # Save the patch to the specified location
                        nib.save(patch_nifti_img_left, patch_img_filepath_left)
                        nib.save(patch_nifti_img_right, patch_img_filepath_right)

# uses lists of axial images and segmentations to extract patches for each disc
def extract_and_save_axial_patches(axial_images, axial_segmentations, nii_folder, output_folder):
    # Match each axial image to its corresponding sagittal segmentation
    for img_name, seg_sag_name in zip(axial_images, axial_segmentations):
        if "patch" not in img_name:
            img_path = os.path.join(nii_folder, img_name)
            seg_sag_path = os.path.join(nii_folder, seg_sag_name)
            
            # Load the volumetric image and sagittal segmentation
            vol = nib.load(img_path).get_fdata()
            seg_sag = nib.load(seg_sag_path).get_fdata()
            affine = nib.load(img_path).affine

            # Détection des disques dans la segmentation sagittale
            #The values to check are based on the classes in totalspineseg 
            disc_l5 = np.isin(seg_sag, [100]).astype(int)
            disc_l4 = np.isin(seg_sag, [95]).astype(int)
            disc_l3 = np.isin(seg_sag, [94]).astype(int)
            disc_l2 = np.isin(seg_sag, [93]).astype(int)
            disc_l1 = np.isin(seg_sag, [92]).astype(int)

            
            discs_dict = {
                "L1_L2": disc_l1,
                "L2_L3": disc_l2,
                "L3_L4": disc_l3,
                "L4_L5": disc_l4,
                "L5_S1": disc_l5
            }

            # Extract and save patches for each disc
            for disc_name, disc_mask in discs_dict.items():
                if np.any(disc_mask):  # If the disc is found in the segmentation
                    # Extract the patch using the segmentation mask
                    
                    patch_img = patch_extraction_volume(vol, disc_mask, affine)
                    
                    if patch_img is not None:  # Proceed only if patch extraction was successful

                        # Construct the filename and file path
                        patch_img_filename = f"{img_name[:-7]}_{disc_name}_patch.nii.gz"
                        patch_img_filepath = os.path.join(output_folder, patch_img_filename)
                        
                        # Use the affine from the original volume to create the patch NIfTI image
                        original_affine = nib.load(img_path).affine
                        patch_nifti_img = nib.Nifti1Image(patch_img, affine=original_affine)

                        original_header = nib.load(img_path).header.copy()

                        q_code = int(original_header['qform_code'])
                        s_code = int(original_header['sform_code'])

                        patch_nifti_img.header.set_qform(original_affine, code=q_code)
                        patch_nifti_img.header.set_sform(original_affine, code=s_code)

                        # Save the patch to the specified location
                        nib.save(patch_nifti_img, patch_img_filepath)

# extract patches from the discs in the nii folder, for axial and sagittal patches
def extract_patches_from_discs(nii_folder, output_folder):
    """
    Traverses a folder containing MRIs and associated sagittal segmentations.
    For each axial image and associated sagittal segmentation, extracts patches for discs with labels 206 to 202.
    Saves each patch in the corresponding folder structure within output_folder.

    nii_folder : path to the folder containing MRIs and segmentations
    output_folder : path to the folder where patches will be saved
    """
    axial_images = []
    axial_segmentations = []
    sagittal_T2_segmentations = []
    sagittal_T1_segmentations = []
    sagittal_T1_images = []
    sagittal_T2_images = []

    
    # Traverse files in the nii_folder
    for filename in os.listdir(nii_folder):
        if 'acq-ax' in filename and filename.endswith('.nii.gz') and not filename.endswith('_seg.nii.gz'):          
            axial_images.append(filename)  # Axial images
        elif 'acq-ax' in filename and 'T2w' in filename and 'total_seg.nii.gz' in filename:
            axial_segmentations.append(filename)  # Sagittal segmentations
        #elif 'acq-sag' in filename and 'T2w' in filename and 'total_seg.nii.gz' in filename:
        #    sagittal_T2_segmentations.append(filename)
        elif 'acq-sag' in filename and 'T1w' in filename and 'total_seg.nii.gz' in filename:
            sagittal_T1_segmentations.append(filename)
        #elif 'acq-sag' in filename and 'T2' in filename and filename.endswith('.nii.gz') and not filename.endswith('_seg.nii.gz'):          
        #    sagittal_T2_images.append(filename)
        elif 'acq-sag' in filename and 'T1' in filename and filename.endswith('.nii.gz') and not filename.endswith('_seg.nii.gz'):          
            sagittal_T1_images.append(filename)

    # Sort lists to ensure corresponding order
    axial_segmentations.sort()
    axial_images.sort()
    sagittal_T2_segmentations.sort()
    sagittal_T2_images.sort()
    sagittal_T1_segmentations.sort()
    sagittal_T1_images.sort()
    sagittal_T2_segmentations.sort()
    
    print(len(sagittal_T2_segmentations),len(sagittal_T2_images))
    print(len(sagittal_T1_segmentations),len(sagittal_T1_images))

    extract_and_save_sagittal_patches(sagittal_T2_images, sagittal_T2_segmentations, nii_folder, output_folder)
    extract_and_save_sagittal_patches(sagittal_T1_images, sagittal_T1_segmentations, nii_folder, output_folder)
    extract_and_save_axial_patches(axial_images, axial_segmentations, nii_folder, output_folder)


# function to extract patches from the discs in the nii folder for axial patches
def patch_extraction_volume(vol, mask, affine):
    """
    Extract a 3D patch from an MRI volume with specific real-world dimensions.
    
    Parameters:
    - vol: 3D numpy array representing the volume
    - mask: 3D segmentation mask 
    - affine: Affine matrix from the NIfTI file
    - header: Header from the NIfTI file
    
    Returns:
    - patch: 3D numpy array with specified real-world dimensions
    """
    # Convert mask to tensor for non-zero index extraction
    mask = torch.Tensor(mask)
    nonzero_indices = torch.nonzero(mask)
    
    # Calculate the centroid of the mask
    centroid = nonzero_indices.float().mean(0).numpy().astype(int)
    
    # Get voxel sizes from the affine matrix
    voxel_sizes = np.abs(np.diag(affine)[:3])
    
    # Calculate the number of voxels corresponding to 2.5 cm posterior displacement
    posterior_displacement_cm = 20
    posterior_displacement_voxels = (posterior_displacement_cm / voxel_sizes[1]).astype(int)
    
    # Compute the new centroid with posterior displacement
    # Assuming the third dimension (index 2) is the posterior-anterior axis
    displaced_centroid = centroid.copy()
    displaced_centroid[1] -= posterior_displacement_voxels
    
    # Define desired patch sizes in cm
    patch_sizes_cm = {
        'RL': 60,  # Right-Left 
        'AP': 40,  # Anterior-Posterior
        'SI': 30   # Superior-Inferior
    }
    
    # Calculate patch size in voxels
    patch_sizes_voxels = np.floor(np.array([
        patch_sizes_cm['RL'] / voxel_sizes[0],
        patch_sizes_cm['AP'] / voxel_sizes[1], 
        patch_sizes_cm['SI'] / voxel_sizes[2]
    ])).astype(int)

    # Extract patch
    D, H, W = vol.shape
    half_sizes = patch_sizes_voxels // 2
    
    patch = vol[
        max(0, displaced_centroid[0] - half_sizes[0]):min(D, displaced_centroid[0] + half_sizes[0] + patch_sizes_voxels[0] % 2),
        max(0, displaced_centroid[1] - half_sizes[1]):min(H, displaced_centroid[1] + half_sizes[1] + patch_sizes_voxels[1] % 2),
        max(0, displaced_centroid[2] - half_sizes[2]):min(W, displaced_centroid[2] + half_sizes[2] + patch_sizes_voxels[2] % 2)
    ]

    return patch

def select_best_patches(folder_path):
    discs = ['L1_L2', 'L2_L3', 'L3_L4', 'L4_L5', 'L5_S1']
    disc_patches = {disc: [] for disc in discs}
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.nii.gz') and '_seg' not in filename:
            for disc in discs:
                if f"{disc}_patch" in filename:
                    file_path = os.path.join(folder_path, filename)
                    img = nib.load(file_path)
                    resolution = img.header.get_zooms()
                    voxel_volume = resolution[0] * resolution[1] * resolution[2]
                    
                    # Check for corresponding segmentation file
                    seg_filename = filename.replace('.nii.gz', '_seg.nii.gz')
                    seg_path = os.path.join(folder_path, seg_filename)
                    
                    if os.path.exists(seg_path):
                        disc_patches[disc].append((file_path, seg_path, voxel_volume))
                        
    for disc, patches in disc_patches.items():
        if len(patches) > 1:
            # Sort patches by increasing voxel volume (resolution)
            patches.sort(key=lambda x: x[2])
            
            # Keep the patch with the best resolution
            best_patch = patches[0]
            
            # Remove other patches and their segmentations
            for patch in patches[1:]:
                os.remove(patch[0])
                os.remove(patch[1])



def process_all_subjects_in_directory(root_dir, output_root_dir):
    """
    Traverses all subdirectories in the root directory corresponding to subjects,
    and applies the patch extraction function to each subdirectory.
    
    root_dir : root directory containing subject subdirectories
    output_root_dir : root directory where output patches are stored
    """

    sub_treated = 0
    sub_failed = 0

    for subject_folder in os.listdir(root_dir):
        
        subject_path = os.path.join(root_dir, subject_folder, "anat")
        output_subject_path = os.path.join(output_root_dir, subject_folder, "anat")
        
        # Check if it is a subdirectory
        if os.path.isdir(subject_path):
            os.makedirs(output_subject_path, exist_ok=True)
            try:
                # Extract patches from discs
                extract_patches_from_discs(subject_path, output_subject_path)
                
                # Select the best patches if there are multiple ones for the same disc
                select_best_patches(output_subject_path)

                print(f"Processed subject {subject_folder}")

                sub_treated += 1

            except Exception as e:
                # Print a message indicating that an exception was raised
                print("An exception was raised during patch processing.")
                
                # Print the arguments passed to 'extract_patches_from_discs'
                print(f"Arguments for 'extract_patches_from_discs': subject_path={subject_path}, output_subject_path={output_subject_path}")
                
                # Print the arguments passed to 'select_best_patches'
                print(f"Arguments for 'select_best_patches': output_subject_path={output_subject_path}")
                
                # Print the type of exception and its details
                print(f"Error type: {type(e).__name__}, Details: {e}")

                sub_failed += 1
    
    print(f"Processed {sub_treated} subjects, {sub_failed} subjects failed")


class ExtractSlicesD(Transform):
    def __init__(self, keys=['image'], target_size=(384, 384), verbose=False):
        self.keys = keys
        self.target_size = target_size
        self.resize = tio.Resize(target_shape=(*target_size, 1))
        self.verbose = verbose

    def __call__(self, data):
        d = dict(data)
        
        for key in self.keys:
            # Get image and remove channel dimension (1, X, Y, 6) -> (X, Y, 6)
            image = d[key].squeeze(0)
            for i in range(image.shape[2]):
                # Extract slice, add channel dim for torchio,
                # resize, then normalize
                slice_2d = image[:, :, i]
                slice_3d = slice_2d.unsqueeze(0).unsqueeze(-1)
                if self.verbose:
                    print(f"Shape before resize: {slice_3d.shape}")
                slice_resized = self.resize(slice_3d)
                if self.verbose:
                    print(f"Shape after resize: {slice_resized.shape}")
                # Remove the z dimension that we added
                slice_final = slice_resized.squeeze(-1)
                d[f'slice_{i}'] = slice_final
                if self.verbose:
                    print(f"Final slice {i} shape: {slice_final.shape}")
        return d

class ExtractSlicesD_nfn(Transform):
    def __init__(self, keys=['image'], target_size=(384, 384), verbose=False):
        self.keys = keys
        self.target_size = target_size
        self.resize = tio.Resize(target_shape=(*target_size, 1))
        self.verbose = verbose

    def __call__(self, data):
        d = dict(data)
        
        for key in self.keys:
            # Get image and remove channel dimension (1, X, Y, 6) -> (X, Y, 6)
            image = d[key].squeeze(0)
            for i in range(image.shape[0]):
                # Extract slice, add channel dim for torchio,
                # resize, then normalize
                slice_2d = image[i, :, :]
                slice_3d = slice_2d.unsqueeze(0).unsqueeze(-1)
                if self.verbose:
                    print(f"Shape before resize: {slice_3d.shape}")
                slice_resized = self.resize(slice_3d)
                if self.verbose:
                    print(f"Shape after resize: {slice_resized.shape}")
                # Remove the z dimension that we added
                slice_final = slice_resized.squeeze(-1)
                d[f'slice_{i}'] = slice_final
                if self.verbose:
                    print(f"Final slice {i} shape: {slice_final.shape}")
        return d




def get_transforms_sas(mode='basic', side='left'):

    # regular transforms just creating the dataset
    regular_transforms = Compose([
            LoadImaged(keys=['image']),
            EnsureChannelFirstd(keys=["image"]),
        Spacingd(keys=['image'], pixdim=(0.4, 0.4, 4.4), mode=('bilinear'))  # Ré-échantillonnage de l'image
        ])
    
    regular_transforms_flip = Compose([
        LoadImaged(keys=['image']),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(keys=['image'], pixdim=(0.4, 0.4, 4.4), mode=('bilinear')),  # Ré-échantillonnage de l'image
        Flipd(keys=['image'], spatial_axis=0)
    ])

    if side == 'left':
        reg_t = regular_transforms
    elif side == 'right':
        reg_t = regular_transforms_flip
    
    if mode == 'basic':
        common_transforms = Compose([
            SpatialCropd(keys=['image'], roi_start=(0, 0, 0), roi_end=(80, 100, 6)),  # crop pour récupérer la gauche
            SpatialPadd(keys=['image'], spatial_size=(60, 80, 6)),  # Padding pour atteindre une taille fixe
            CenterSpatialCropd(keys=['image'], roi_size=(60, 80, 6))  # Crop pour obtenir une taille fixe
        ])

    # Create list of transforms for processing 2D slices
    slice_transforms = Compose([
        # Custom transform to extract and resize slices
        ExtractSlicesD(keys=['image'], target_size=(384, 384)),
        # Scale and normalize
        ScaleIntensityd(
            keys=[f'slice_{i}' for i in range(6)]
        ),
        NormalizeIntensityd(
            keys=[f'slice_{i}' for i in range(6)],
            nonzero=True
        ),
        # Ensure all slices are tensors
        ToTensord(
            keys=[f'slice_{i}' for i in range(6)]
        ),
        # Concatenate all slices into a bag
        ConcatItemsd(
            keys=[f'slice_{i}' for i in range(6)],
            name='bag',
            dim=0
        ),
        # Add a transform to ensure bag has the correct shape
        Lambdad(
            keys=['bag'],
            func=lambda x: x.reshape(6, 1, 384, 384)
        )
    ])

    # Combine common_transforms with slice_transforms
    transforms = Compose([reg_t, common_transforms, slice_transforms])

    return transforms


def prepare_data_sas(list_subjects, data_dir, transform_left, transform_right):
    data_right = []
    data_left = []
    
    counter = 0
    
    # Dictionnaire de conversion des étiquettes
    text2int = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}
    
    for subject in list_subjects:
        
        subject_dir = os.path.join(data_dir, f'sub-{subject}', 'anat')
        
        if os.path.isdir(subject_dir):
            for file in os.listdir(subject_dir):
                
                 if '_patch.nii.gz' in file and 'foramen' not in file:
                    image_path = os.path.join(subject_dir, file)
                    
                    parts = image_path.split('_')
                    disk_level = f"{parts[-3]}_{parts[-2]}"

                    if os.path.exists(image_path):
                        
                        subject_id = (subject.replace('sub-', ''))
                        
                        label_column = f'_subarticular_stenosis_{disk_level.lower()}'
                        
                        
                        label_left = f"{subject_id}_left{label_column}"
                        label_right = f"{subject_id}_right{label_column}"
                        
                        
                        data_right.append({"image": image_path, "label": label_right})
                        data_left.append({"image": image_path, "label": label_left})
                        counter += 2

    print(f"Nombre de données chargées: {counter}")
    
    return ConcatDataset([Dataset(data=data_left, transform=transform_left), Dataset(data=data_right, transform=transform_right)])



                           


def get_transforms_scs(mode='basic'):


    regular_transforms = Compose([
        LoadImaged(keys=['image']),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(keys=['image'], pixdim=(0.4, 0.4, 4.4), mode=('bilinear')),  # Ré-échantillonnage de l'image
    ])
    
    if mode == 'basic':
        common_transforms = Compose([
            SpatialPadd(keys=['image'], spatial_size=(120, 80, 6)),
            CenterSpatialCropd(
                keys=['image'],
                roi_size=(120, 80, 6)
            ),
        ])

    elif mode == 'random':
        # Same transforms but with random augmentations
        common_transforms = Compose([
            RandRotated(keys=['image'], prob=1.0, range_x=0.2),
            RandAffined(keys=['image'], prob=1.0, shear_range=(0.3, 0.3, 0.3)),
            Rand3DElasticd(keys=['image'], prob=0.5, sigma_range=(8, 12), magnitude_range=(100, 200)),
            RandGaussianNoised(keys=['image'], prob=0.5, mean=0.0, std=0.1),
            RandBiasFieldd(keys=['image'], prob=0.5, coeff_range=(0, 0.4)),
            RandZoomd(keys=['image'], prob=0.5, min_zoom=0.95, max_zoom=1.15),
            SpatialPadd(keys=['image'], spatial_size=(120, 80, 6)),
            RandSpatialCropd(
                keys=['image'],
                roi_size=(120, 80, 6),
                random_size=False
            ),
        ])

    # Create list of transforms for processing 2D slices
    slice_transforms = Compose([
        # Custom transform to extract and resize slices
        ExtractSlicesD(keys=['image'], target_size=(384, 384)),
        # Scale and normalize
        ScaleIntensityd(
            keys=[f'slice_{i}' for i in range(6)]
        ),
        NormalizeIntensityd(
            keys=[f'slice_{i}' for i in range(6)],
            nonzero=True
        ),
        # Ensure all slices are tensors
        ToTensord(
            keys=[f'slice_{i}' for i in range(6)]
        ),
        # Concatenate all slices into a bag
        ConcatItemsd(
            keys=[f'slice_{i}' for i in range(6)],
            name='bag',
            dim=0
        ),
        # Add a transform to ensure bag has the correct shape
        Lambdad(
            keys=['bag'],
            func=lambda x: x.reshape(6, 1, 384, 384)
        )
    ])

    # Combine common_transforms with slice_transforms
    transforms = Compose([regular_transforms, common_transforms, slice_transforms])

    return transforms



def prepare_data_scs(list_subjects, data_dir, transform):
    data = []
    
    counter = 0

    # Dictionnaire de conversion des étiquettes
    text2int = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}
    
    for subject in list_subjects:
        
        subject_dir = os.path.join(data_dir, f'sub-{subject}', 'anat')
        if os.path.isdir(subject_dir):
            for file in os.listdir(subject_dir):
                
                if '_patch.nii.gz' in file and 'foramen' not in file:
                    image_path = os.path.join(subject_dir, file)
                    
                    parts = image_path.split('_')
                    disk_level = f"{parts[-3]}_{parts[-2]}"

                    if os.path.exists(image_path):
                        
                        
                        subject_id = (subject.replace('sub-', ''))
                        
                        label_column = f'spinal_canal_stenosis_{disk_level.lower()}'
                        
                         
                        
                        counter += 1
                        label = f"{subject_id}_{label_column}"
                        data.append({"image": image_path, "label": label})


    print(f"Nombre de données chargées: {counter}")
    return Dataset(data=data, transform=transform)


def get_transforms_nfn(mode='basic', side = "right"):

    regular_transforms = Compose([
        LoadImaged(keys=['image']),
        EnsureChannelFirstd(keys=["image"]),
    ])

    if side == "left": 
        regular_transforms = Compose([regular_transforms,Flipd(keys=['image'], spatial_axis=0)])


    if mode == 'basic':
        common_transforms = Compose([
            Spacingd(keys=['image'], pixdim=(4.0, 0.4, 0.4), mode=('bilinear')),
            SpatialPadd(keys=['image'], spatial_size=(6, 100, 100)),
            CenterSpatialCropd(
                keys=['image'],
                roi_size=(6, 100, 100)
            ),
        ])

    elif mode == 'random':
        # Same transforms but with random augmentations
        common_transforms = Compose([
            #Spacingd(keys=['image'], pixdim=(4.0, 0.4, 0.4), mode=('bilinear')),
            RandRotated(keys=['image'], prob=0.8, range_y=0.2),
            RandGaussianNoised(keys=['image'], prob=0.4, mean=0.0, std=0.1),
            RandBiasFieldd(keys=['image'], prob=0.4, coeff_range=(0, 0.3)),
            SpatialPadd(keys=['image'], spatial_size=(6, 100, 100)),
            RandSpatialCropd(
                keys=['image'],
                roi_size=(6, 100, 100),
                random_size=False
            ),
        ])

    # Create list of transforms for processing 2D slices
    slice_transforms = Compose([
        # Custom transform to extract and resize slices
        ExtractSlicesD_nfn(keys=['image'], target_size=(224, 224)),
        # Scale and normalize
        ScaleIntensityd(
            keys=[f'slice_{i}' for i in range(6)]
        ),
        NormalizeIntensityd(
            keys=[f'slice_{i}' for i in range(6)],
            nonzero=True
        ),
        # Ensure all slices are tensors
        ToTensord(
            keys=[f'slice_{i}' for i in range(6)]
        ),
        # Concatenate all slices into a bag
        ConcatItemsd(
            keys=[f'slice_{i}' for i in range(6)],
            name='bag',
            dim=0
        ),
        # Add a transform to ensure bag has the correct shape
        Lambdad(
            keys=['bag'],
            func=lambda x: x.reshape(6, 1, 224, 224)
        )
    ])

    # Combine common_transforms with slice_transforms
    transforms = Compose([regular_transforms, common_transforms, slice_transforms])

    return transforms


def prepare_data_nfn(list_subject, data_dir, random=False):
    data_right = []
    data_left = []
    

    counter = 0
    # Label conversion dictionary
    text2int = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}

    for subject in list_subjects:
        
        subject_dir = os.path.join(data_dir, f'sub-{subject}', 'anat')
        if os.path.isdir(subject_dir):
            for file in os.listdir(subject_dir):
                if '_patch.nii.gz' in file and 'foramen' in file and 'T1' in file:
                    image_path = os.path.join(subject_dir, file)
                    parts = image_path.split('_')
                    disk_level = f"{parts[-5]}_{parts[-4]}"

                    if os.path.exists(image_path):
                        
                        subject_id = subject
                        if 'left' in file:
                            orientation = 'right'
                        elif 'right' in file: 
                            orientation = 'left'
                        label_column = (
                            f'{orientation}_neural_foraminal_narrowing_{disk_level.lower()}'
                        )
                        # Get raw label
                        label = f"{subject_id}_{label_column}"

                        # Convert text label to numeric value
                        
                        counter += 1
                        if "left" in image_path: 
                            data_right.append({
                                "image": image_path,
                                "label": label
                            })
                        if "right" in image_path: 
                            data_left.append({
                                "image": image_path,
                                "label": label
                            })

    print(f"Number of loaded data: {counter}")
    return ConcatDataset([Dataset(data=data_left, transform=get_transforms_nfn(mode='random', side='left') if random else get_transforms_nfn(mode='basic',side='left')), Dataset(data=data_right, transform=get_transforms_nfn(mode='random', side='right') if random else get_transforms_nfn(mode='basic',side='right'))]) 



'''
File to introduce a MIL model
Note that loads of hyperparameters could be included as arguments
It could be avg pooling size, hidden dim, etc...
Also encoder could be changed to a different model
'''

import torch
import torch.nn as nn

# import timm for models
import timm


# define a MIL model
class MILsection(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=1):
        super(MILsection, self).__init__()
        self.num_layers = num_layers
        if num_layers > 0:
            self.rnn = nn.GRU(input_dim, input_dim//2, num_layers=num_layers,
                             batch_first=True, dropout=0.1, bidirectional=True)
        self.aux_attention = nn.Sequential(
            nn.Tanh(),
            nn.Linear(input_dim, 1)
        )
        self.attention = nn.Sequential(
            nn.Tanh(),
            nn.Linear(input_dim, 1)
        )

    def forward(self, bags):
        """
        Args:
            bags: (batch_size, num_instances, input_dim)

        Returns:
            logits: (batch_size, num_classes)
        """
        batch_size, num_instances, input_dim = bags.size()

        if self.num_layers > 0:
            bags_rnn, _ = self.rnn(bags)
        else:
            bags_rnn = bags
        
        # Main attention
        attn_scores = self.attention(bags_rnn).squeeze(-1)  # [batch_size, num_instances]
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [batch_size, num_instances]
        weighted_instances = torch.bmm(attn_weights.unsqueeze(1), bags_rnn).squeeze(1)  # [batch_size, input_dim]
        
        # Auxiliary attention - process each instance independently
        aux_attn_scores = self.aux_attention(bags_rnn).squeeze(-1)  # [batch_size, num_instances]
        aux_features = bags_rnn  # [batch_size, num_instances, input_dim]
        
        return weighted_instances, aux_features


# here define the whole MIL model
# uses the MILsection model and a ConvNext Small as a feature extractor
# note that loads of hyperparameters could be included as arguments
class MILmodel(nn.Module):
    def __init__(self, encoder, num_layers=1):
        super(MILmodel, self).__init__()
        # encoder
        self.encoder = encoder
        # flattening layer, applying pooling and flattening
        # note here that we could try different pooling methods
        self.flatten = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1)
        )
        self.feature_size = self.encoder.num_features
        
        # MIL section, loads of hyperparameters here also
        self.mil_section = MILsection(input_dim=self.feature_size,
                                    hidden_dim=1024, 
                                    num_classes=3,
                                    num_layers=num_layers)
        # classifier output
        self.classifier = nn.Linear(self.feature_size, 3)
        # aux classifier output - now takes each instance independently
        self.aux_classifier = nn.Linear(self.feature_size, 3)

    def forward(self, x):
        # x shape: (batch_size, 6, 1, 384, 384)
        batch_size, num_instances, channels, H, W = x.shape

        # Reshape to process all instances through encoder
        x = x.reshape(-1, channels, H, W)  # shape: (batch_size * 6, 1, 384, 384)
        
        # Pass through encoder
        x = self.encoder.forward_features(x)  # shape: (batch_size * 6, feature_size, h', w')
        
        # Apply pooling and flatten
        x = self.flatten(x)  # shape: (batch_size * 6, feature_size)
        
        # Reshape back to separate instances
        x = x.reshape(batch_size, num_instances, self.feature_size)  # shape: (batch_size, 6, feature_size)
        
        # Pass through MIL section
        weighted_instances, aux_features = self.mil_section(x)
        # weighted_instances: (batch_size, feature_size)
        # aux_features: (batch_size, num_instances, feature_size)
        
        # Main classification
        main_output = self.classifier(weighted_instances)  # shape: (batch_size, 3)
        
        # Auxiliary classification - apply to each instance independently
        aux_output = self.aux_classifier(aux_features)  # shape: (batch_size, num_instances, 3)
        # Average the auxiliary predictions across instances
        aux_output = aux_output.mean(dim=1)  # shape: (batch_size, 3)
        
        return main_output, aux_output


convnext_small_sas = timm.create_model('convnext_small.fb_in22k_ft_in1k_384',
                                   in_chans=1, pretrained=False, num_classes=0)
convnext_small_scs = timm.create_model('convnext_small.fb_in22k_ft_in1k_384',
                                   in_chans=1, pretrained=False, num_classes=0)
convnext_small_nfn = timm.create_model('convnext_small.fb_in22k_ft_in1k_384',
                                   in_chans=1, pretrained=False, num_classes=0)



def load_model(model_path, layers, patho, device='cuda'):
    if patho == 'sas':
        encoder = convnext_small_sas
    elif patho == 'scs':
        encoder = convnext_small_scs
    elif patho == 'nfn':
        encoder = convnext_small_nfn
        
    """Load the trained MIL model from the checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    model = MILmodel(encoder=encoder, num_layers=layers).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    return model



model_sas = load_model('/kaggle/input/sas_mil/other/default/2/best_mil_model.pth',2, 'sas')


model_scs = load_model('/kaggle/input/scs_mil/other/default/3/best_mil_model.pth' ,2, 'scs')


model_nfn = load_model('/kaggle/input/nfn_mil/other/default/5/best_mil_model5861.pth' ,2, 'nfn')


def copy_subject_folders(list_subjects, train, target_dir):
    """
    Copies folders matching the subject names from the source directory to the target directory.

    :param list_subjects: List of subject folder names to copy (e.g., ["01", "02"]).
    :param train: boolean to know if you want to fetch your subjects from train or test dataset
    """
    if train: 
        source_dir = "/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images"
    else: 
        source_dir = "/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images"

   
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for subject in list_subjects:
        subject_path = os.path.join(source_dir, subject)
        target_path = os.path.join(target_dir, subject)
        
        if os.path.exists(subject_path):
            shutil.copytree(subject_path, target_path)
        else:
            print(f"Subject folder {subject} does not exist in {source_dir}")




def preprocessing(list_subjects, train=False): 
    
    ### Copy the dcom files 
    dcom_folder = "/kaggle/working/dcom_data"
    copy_subject_folders(list_subjects, train, dcom_folder)
    
    ### niftification 
    nii_folder = "/kaggle/working/nii_data"

          
        
    os.makedirs(nii_folder, exist_ok=True)

    
    # Process subjects and set up directories
    
    for subject_id in list_subjects:  # Adjust range as needed: 1975 subjects
        try: 
        
            # Create specific directories
            os.makedirs(os.path.join(nii_folder, f'sub-{subject_id}'), exist_ok=True)
            os.makedirs(os.path.join(nii_folder, f'sub-{subject_id}', 'anat'), exist_ok=True)

            # Process subject and set up directories
            process_subject(subject_id, input_folder, nii_folder, df_meta_f, meta_obj)
        except: 
            print(f'failed preprocessing for {subject_id}')
 

    for item in os.listdir(nii_folder):
        item_path = os.path.join(nii_folder, item)
        
        # Check if the item is a directory and starts with "sub"
        if os.path.isdir(item_path) and item.startswith("sub"):
            continue  # Skip deletion for folders starting with "sub"
        
        # Delete the item (file or directory)
        if os.path.isfile(item_path):
            os.remove(item_path)
            print(f"Deleted file: {item_path}")
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)
            print(f"Deleted folder: {item_path}")

    ### Totalspineseg 
    os.system("cp -r /kaggle/input/nnunet_totalspineseg/other/20250108_totalspineseg_version/10/nnUNet /kaggle/working/TotalSpineSeg/tss_input")

    run_totalspineseg(nii_folder)
    
    ### Patch extraction 
    process_directory_other(nii_folder)
    process_all_subjects_in_directory(nii_folder, nii_folder)



def eval_sas(list_subjects, data_dir):     
    # Préparer les données
    transform_left=get_transforms_sas(mode='basic',side='left')
    transform_right=get_transforms_sas(mode='basic',side='right')
    data = prepare_data_sas(list_subjects, data_dir, transform_left=transform_left, transform_right=transform_right)
    data_loader = DataLoader(data, batch_size=16)
    
    pred = []
    i=0
    
    with torch.no_grad():
            for batch in tqdm(data_loader):
                
                
                inputs = batch["bag"].cuda()
                labels = batch["label"]
                
                main, aux = model_sas(inputs)

                '''if i<3:
                    print("VISUALIZING SAS BATCH")
                    visualize_batch(batch)
                    i+=1'''
                probs = torch.softmax(main, dim=1).cpu().numpy()
                #adjusted_probs = probs.copy()
                #max_class_indices = np.argmax(probs, axis=1)
                #boost_mask = (max_class_indices == 2)
            
                # Appliquer le boost uniquement sur les lignes concernées
                #adjusted_probs[boost_mask, 2] *= 1.3
            
                # Renormalisation
                #row_sums = adjusted_probs.sum(axis=1, keepdims=True)
                #normalized_probs = adjusted_probs / row_sums

                #boost_mask_mid = (max_class_indices == 1)
            
                # Appliquer le boost uniquement sur les lignes concernées
                #normalized_probs[boost_mask, 1] *= 1.15
            
                # Renormalisation
                #row_sums_1 = normalized_probs.sum(axis=1, keepdims=True)
                #normalized_probs2 = normalized_probs / row_sums_1

                # change for normalized probs to apply boosting
                outputs = list(probs)
                
                for i in range(len(labels)): 
                    label = labels [i]
                    output = list(outputs[i])
                    pred.append((label, output))
        
    return pred 
        


def visualize_batch(batch):
    """
    Visualize a batch of images and save them to wandb
    batch: dictionary containing 'bag' tensor of shape [B, 6, 1, 384, 384] and 'label'
    epoch: current epoch number
    """
    # Get the first batch and ensure it's on CPU
    images = batch['bag'].cpu().detach()  # Shape: [B, 6, 1, 384, 384]
    labels = batch['label']
    
    # Take only the first 4 samples to avoid too large figures
    n_samples = min(4, images.shape[0])
    
    # Create a figure with subplots for each sample and its 6 slices
    fig, axes = plt.subplots(n_samples, 6, figsize=(20, 4*n_samples))
    if n_samples == 1:
        axes = axes[None, :]  # Add dimension for consistent indexing
    
    for i in range(n_samples):
        for j in range(6):
            # Get the image slice and ensure it's a valid image
            img = images[i, j, 0].numpy()
            
            # Normalize the image for better visualization
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            
            # Plot the image
            axes[i, j].imshow(img, cmap='gray')
            axes[i, j].axis('off')
            
            # Add title only to the first row
            if i == 0:
                axes[i, j].set_title(f'Slice {j+1}')
    
    plt.tight_layout()
    plt.show()



def eval_scs(list_subjects, data_dir):     
    # Préparer les données
    transform=get_transforms_scs(mode='basic')
    data = prepare_data_scs(list_subjects, data_dir, transform)
    data_loader = DataLoader(data, batch_size=16)
    pred = []
    i = 0
    
    with torch.no_grad():
            for batch in tqdm(data_loader):


                
                inputs = batch["bag"].cuda()
                labels = batch["label"]
                '''if i<3:
                    print("VISUALIZING SCS BATCH")
                    visualize_batch(batch)
                    i+=1'''
                main, aux = model_scs(inputs)

                out_arr = torch.softmax(main, dim=1).cpu().numpy()
                #out_arr[:, 2] *= 1.25  # Multiplie la proba de la 3ème classe
                # Renormalisation
                #row_sums = out_arr.sum(axis=1, keepdims=True)
                #out_norm = out_arr / row_sums

                # change for out_norm to apply severe change
                outputs = list(out_arr)
                
                for i in range(len(labels)): 
                    label = labels [i]
                    output = list(outputs[i])
                    pred.append((label, output))
        
    return pred 
 
        


def eval_nfn(list_subjects, data_dir): 
    
        # Préparer les données
        transform=get_transforms_nfn()
        data = prepare_data_nfn(list_subjects, data_dir, random=False)
        data_loader = DataLoader(data, batch_size=16)
    
        model_nfn.eval()
        
        pred = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader):
                visualize_batch(batch)
                
                inputs = batch["bag"].cuda()
                labels = batch["label"]
                
                main, aux = model_scs(inputs)

                out_arr = torch.softmax(main, dim=1).cpu().numpy()
                
                outputs = list(out_arr)
                
                for i in range(len(labels)): 
                    label = labels [i]
                    output = list(outputs[i])
                    pred.append((label, output))
        return pred 


def eval(list_subjects, data_dir): 


    #pred_nfn = eval_nfn(list_subjects, data_dir)
    pred_scs = eval_scs(list_subjects, data_dir)
    #pred_sas = eval_sas(list_subjects, data_dir)

    #for label, output in pred_nfn: 
        #result.loc[result["row_id"] == label, ['normal_mild', 'moderate', 'severe']] = output
    for label, output in pred_scs: 
        result.loc[result["row_id"] == label, ['normal_mild', 'moderate', 'severe']] = output
    #for label, output in pred_sas: 
        #result.loc[result["row_id"] == label, ['normal_mild', 'moderate', 'severe']] = output
   

    


def inference(list_subjects): 
    list_subjects = list(map(str, list_subjects))
    preprocessing(list_subjects)
    data_dir = "/kaggle/working/nii_data"
    eval(list_subjects, data_dir)
    shutil.rmtree("/kaggle/working/nii_data")
    shutil.rmtree("/kaggle/working/dcom_data")
    


if train: 
        csv_description = "/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_series_descriptions.csv"
        input_folder = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images'
else: 
        csv_description = "/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_series_descriptions.csv"
        input_folder = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images'


df_meta_f = pd.read_csv(csv_description, sep=',')
subject_ids = np.unique(df_meta_f["study_id"].values)

# List out all of the Studies we have on patients.
part_1 = os.listdir(input_folder)
part_1 = list(filter(lambda x: x.find('.DS') == -1, part_1))

p1 = [(x, f"{input_folder}/{x}") for x in part_1]
meta_obj = { p[0]: { 'folder_path': p[1], 
                    'SeriesInstanceUIDs': [] 
                } 
            for p in p1 }

for m in meta_obj:
    meta_obj[m]['SeriesInstanceUIDs'] = list( 
        filter(lambda x: x.find('.DS') == -1, 
            os.listdir(meta_obj[m]['folder_path'])
            )
    )
# Grabs the corresponding series descriptions
for k in tqdm(meta_obj):
    for s in meta_obj[k]['SeriesInstanceUIDs']:
        if 'SeriesDescriptions' not in meta_obj[k]:
            meta_obj[k]['SeriesDescriptions'] = []
        try:
            meta_obj[k]['SeriesDescriptions'].append(
                df_meta_f[(df_meta_f['study_id'] == int(k)) & 
                (df_meta_f['series_id'] == int(s))]['series_description'].iloc[0])
        except:
            None


df = pd.read_csv('/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/sample_submission.csv')
df['study_id'] = df.row_id.str.extract(r'([0-9]*)_').astype('int')
df['side'] = df.row_id.str.extract(r'[0-9]_([a-z]*)_')
df['level'] = df.row_id.str.extract(r'([a-z0-9]*)_[a-z0-9]*$')
df.side = df.side.map({'left':0,'right':1,'spinal':2})
df.level = df.level.map({'l1':0,'l2':1,'l3':2,'l4':3,'l5':4})
result = df[['row_id']]
result.loc[:, ['normal_mild', 'moderate', 'severe']] = 1/3


result


shutil.rmtree("/kaggle/working/nii_data")
shutil.rmtree("/kaggle/working/dcom_data")


subject_ids = np.unique(df_meta_f["study_id"].values).tolist()

B = 48
N = len(subject_ids)
Q, R = N//B, N%B

for i in range(Q):
    
    list_subjects = subject_ids[i*B:(i+1)*B]
 
    inference(list_subjects)
    
    
if R!=0:
    list_subjects = subject_ids[Q*B:]
    inference(list_subjects)


sub = result.copy()
sub = sub[['row_id','normal_mild','moderate','severe']]
sub.to_csv('submission.csv',index=False)


sub


# Cleaning working folder

listdir = os.listdir("/kaggle/working")
for pth in listdir:
    if pth != "submission.csv":
        if os.path.isdir("/kaggle/working/"+pth):
            os.system("rm -r /kaggle/working/"+pth)
        else:
            os.system("rm /kaggle/working/"+pth)







