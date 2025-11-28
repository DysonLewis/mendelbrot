#include <Python.h>
#include <numpy/arrayobject.h>
#include <cmath>
#include <algorithm>
#include <cstring>

// Lanczos kernel function for high-quality downsampling
inline double lanczos_kernel(double x, int a = 3) {
    if (x == 0.0) return 1.0;
    if (x < -a || x > a) return 0.0;
    
    double pi_x = M_PI * x;
    double pi_x_a = pi_x / a;
    return (a * std::sin(pi_x) * std::sin(pi_x_a)) / (pi_x * pi_x_a);
}

// Downsample a 512x512x3 image to 256x256x3 using Lanczos interpolation
static PyObject* downsample_tile(PyObject* self, PyObject* args) {
    PyArrayObject *input_array;
    
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &input_array)) {
        return NULL;
    }
    
    if (PyArray_NDIM(input_array) != 3) {
        PyErr_SetString(PyExc_ValueError, "Input must be 3D array (height, width, channels)");
        return NULL;
    }
    
    npy_intp* dims = PyArray_DIMS(input_array);
    npy_intp src_h = dims[0];
    npy_intp src_w = dims[1];
    npy_intp channels = dims[2];
    
    if (channels != 3) {
        PyErr_SetString(PyExc_ValueError, "Input must have 3 channels (RGB)");
        return NULL;
    }
    
    // Output dimensions are half of input
    npy_intp dst_h = src_h / 2;
    npy_intp dst_w = src_w / 2;
    npy_intp out_dims[3] = {dst_h, dst_w, 3};
    
    PyArrayObject* result = (PyArrayObject*)PyArray_SimpleNew(3, out_dims, NPY_UINT8);
    if (result == NULL) {
        return NULL;
    }
    
    unsigned char* src_data = (unsigned char*)PyArray_DATA(input_array);
    unsigned char* dst_data = (unsigned char*)PyArray_DATA(result);
    
    // Lanczos-3 downsampling
    const int a = 3;
    const double scale = 2.0;
    
    for (npy_intp dst_y = 0; dst_y < dst_h; dst_y++) {
        for (npy_intp dst_x = 0; dst_x < dst_w; dst_x++) {
            // Map destination pixel to source coordinates
            double src_y_center = (dst_y + 0.5) * scale;
            double src_x_center = (dst_x + 0.5) * scale;
            
            double pixel_sum[3] = {0.0, 0.0, 0.0};
            double weight_sum = 0.0;
            
            // Sample from a window around the center
            npy_intp y_start = std::max(0L, (npy_intp)std::ceil(src_y_center - a));
            npy_intp y_end = std::min(src_h, (npy_intp)std::floor(src_y_center + a) + 1);
            npy_intp x_start = std::max(0L, (npy_intp)std::ceil(src_x_center - a));
            npy_intp x_end = std::min(src_w, (npy_intp)std::floor(src_x_center + a) + 1);
            
            for (npy_intp src_y = y_start; src_y < y_end; src_y++) {
                double dy = src_y - src_y_center;
                double wy = lanczos_kernel(dy, a);
                
                for (npy_intp src_x = x_start; src_x < x_end; src_x++) {
                    double dx = src_x - src_x_center;
                    double wx = lanczos_kernel(dx, a);
                    double weight = wx * wy;
                    
                    npy_intp src_idx = (src_y * src_w + src_x) * 3;
                    
                    for (int c = 0; c < 3; c++) {
                        pixel_sum[c] += src_data[src_idx + c] * weight;
                    }
                    weight_sum += weight;
                }
            }
            
            npy_intp dst_idx = (dst_y * dst_w + dst_x) * 3;
            for (int c = 0; c < 3; c++) {
                double value = pixel_sum[c] / weight_sum;
                dst_data[dst_idx + c] = (unsigned char)std::clamp(value, 0.0, 255.0);
            }
        }
    }
    
    return (PyObject*)result;
}

// Assemble a horizontal strip from multiple chunks
static PyObject* assemble_strip(PyObject* self, PyObject* args) {
    PyObject *chunk_list;
    PyObject *positions_list;
    int strip_height, total_width;
    
    if (!PyArg_ParseTuple(args, "OOii", &chunk_list, &positions_list, &strip_height, &total_width)) {
        return NULL;
    }
    
    if (!PyList_Check(chunk_list) || !PyList_Check(positions_list)) {
        PyErr_SetString(PyExc_ValueError, "Arguments must be lists");
        return NULL;
    }
    
    Py_ssize_t n_chunks = PyList_Size(chunk_list);
    if (n_chunks != PyList_Size(positions_list)) {
        PyErr_SetString(PyExc_ValueError, "Chunk list and positions list must have same length");
        return NULL;
    }
    
    npy_intp out_dims[3] = {strip_height, total_width, 3};
    PyArrayObject* result = (PyArrayObject*)PyArray_SimpleNew(3, out_dims, NPY_UINT8);
    if (result == NULL) {
        return NULL;
    }
    
    unsigned char* dst_data = (unsigned char*)PyArray_DATA(result);
    memset(dst_data, 0, strip_height * total_width * 3);
    
    for (Py_ssize_t i = 0; i < n_chunks; i++) {
        PyArrayObject* chunk = (PyArrayObject*)PyList_GetItem(chunk_list, i);
        PyObject* pos_tuple = PyList_GetItem(positions_list, i);
        
        long x_start, x_end;
        if (!PyArg_ParseTuple(pos_tuple, "ll", &x_start, &x_end)) {
            Py_DECREF(result);
            return NULL;
        }
        
        if (PyArray_NDIM(chunk) != 3) {
            PyErr_SetString(PyExc_ValueError, "Chunks must be 3D arrays");
            Py_DECREF(result);
            return NULL;
        }
        
        npy_intp* chunk_dims = PyArray_DIMS(chunk);
        npy_intp chunk_height = chunk_dims[0];
        npy_intp chunk_width = chunk_dims[1];
        
        if (chunk_height != strip_height || chunk_width != (x_end - x_start)) {
            PyErr_SetString(PyExc_ValueError, "Chunk dimensions don't match expected size");
            Py_DECREF(result);
            return NULL;
        }
        
        unsigned char* src_data = (unsigned char*)PyArray_DATA(chunk);
        
        for (npy_intp y = 0; y < chunk_height; y++) {
            npy_intp dst_offset = (y * total_width + x_start) * 3;
            npy_intp src_offset = y * chunk_width * 3;
            memcpy(dst_data + dst_offset, src_data + src_offset, chunk_width * 3);
        }
    }
    
    return (PyObject*)result;
}

// Flip image vertically in-place
static PyObject* flip_vertical(PyObject* self, PyObject* args) {
    PyArrayObject *array;
    
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &array)) {
        return NULL;
    }
    
    if (PyArray_NDIM(array) != 3) {
        PyErr_SetString(PyExc_ValueError, "Input must be 3D array");
        return NULL;
    }
    
    npy_intp* dims = PyArray_DIMS(array);
    npy_intp height = dims[0];
    npy_intp width = dims[1];
    npy_intp channels = dims[2];
    
    unsigned char* data = (unsigned char*)PyArray_DATA(array);
    npy_intp row_size = width * channels;
    
    unsigned char* temp_row = new unsigned char[row_size];
    
    for (npy_intp y = 0; y < height / 2; y++) {
        npy_intp top_offset = y * row_size;
        npy_intp bottom_offset = (height - 1 - y) * row_size;
        
        memcpy(temp_row, data + top_offset, row_size);
        memcpy(data + top_offset, data + bottom_offset, row_size);
        memcpy(data + bottom_offset, temp_row, row_size);
    }
    
    delete[] temp_row;
    
    Py_INCREF(array);
    return (PyObject*)array;
}

// Split a strip into tiles
static PyObject* split_into_tiles(PyObject* self, PyObject* args) {
    PyArrayObject *strip_array;
    int tile_size;
    
    if (!PyArg_ParseTuple(args, "O!i", &PyArray_Type, &strip_array, &tile_size)) {
        return NULL;
    }
    
    if (PyArray_NDIM(strip_array) != 3) {
        PyErr_SetString(PyExc_ValueError, "Input must be 3D array");
        return NULL;
    }
    
    npy_intp* dims = PyArray_DIMS(strip_array);
    npy_intp strip_height = dims[0];
    npy_intp strip_width = dims[1];
    
    int tiles_wide = (strip_width + tile_size - 1) / tile_size;
    
    PyObject* tile_list = PyList_New(tiles_wide);
    if (tile_list == NULL) {
        return NULL;
    }
    
    unsigned char* strip_data = (unsigned char*)PyArray_DATA(strip_array);
    
    for (int tile_idx = 0; tile_idx < tiles_wide; tile_idx++) {
        npy_intp tile_x_start = tile_idx * tile_size;
        npy_intp tile_x_end = std::min(tile_x_start + tile_size, strip_width);
        npy_intp tile_width = tile_x_end - tile_x_start;
        
        npy_intp tile_dims[3] = {tile_size, tile_size, 3};
        PyArrayObject* tile = (PyArrayObject*)PyArray_SimpleNew(3, tile_dims, NPY_UINT8);
        if (tile == NULL) {
            Py_DECREF(tile_list);
            return NULL;
        }
        
        unsigned char* tile_data = (unsigned char*)PyArray_DATA(tile);
        memset(tile_data, 0, tile_size * tile_size * 3);
        
        for (npy_intp y = 0; y < strip_height; y++) {
            npy_intp src_offset = (y * strip_width + tile_x_start) * 3;
            npy_intp dst_offset = y * tile_size * 3;
            memcpy(tile_data + dst_offset, strip_data + src_offset, tile_width * 3);
        }
        
        PyList_SetItem(tile_list, tile_idx, (PyObject*)tile);
    }
    
    return tile_list;
}

static PyMethodDef ImageProcessorMethods[] = {
    {"downsample_tile", downsample_tile, METH_VARARGS,
     "Downsample a tile using Lanczos interpolation"},
    {"assemble_strip", assemble_strip, METH_VARARGS,
     "Assemble a horizontal strip from chunks"},
    {"flip_vertical", flip_vertical, METH_VARARGS,
     "Flip an image vertically in-place"},
    {"split_into_tiles", split_into_tiles, METH_VARARGS,
     "Split a strip into tiles"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef imageprocessormodule = {
    PyModuleDef_HEAD_INIT,
    "image_processor",
    "Fast image processing operations in C++",
    -1,
    ImageProcessorMethods
};

PyMODINIT_FUNC PyInit_image_processor(void) {
    import_array();
    return PyModule_Create(&imageprocessormodule);
}