#include <Python.h>
#include <numpy/arrayobject.h>
#include <cmath>

/*
After writing this I realized that the instruction said
to use the existing code in HW1, which is funny since i Based this off said code
I should have read that line...
I guess I'll keep using this though lol
*/

// Mandelbrot calculation constants
const int MAX_ITER = 100;
const double R2_MAX = 262144.0;
const double LOG2 = std::log(2.0);

// Compute Mandelbrot iteration value for a single point
inline double calc_mandelbrot(double x0, double y0) {
    int ii = 0;
    double x = 0.0, y = 0.0;
    
    // Iterate until escape or max iterations
    while ((x*x + y*y <= R2_MAX) && (ii < MAX_ITER)) {
        double xt = x*x - y*y + x0;
        y = 2*x*y + y0;
        x = xt;
        ii++;
    }
    
    // Apply smooth coloring adjustment if escaped
    if (ii < MAX_ITER) {
        double log_zn = std::log(x*x + y*y) / 2.0;
        double nu = std::log(log_zn / LOG2) / LOG2;
        return ii + 1 - nu;
    }
    
    return ii;
}

// Python-callable function to compute Mandelbrot values for meshgrid arrays
static PyObject* calc_val(PyObject* self, PyObject* args) {
    PyArrayObject *x_array, *y_array;
    
    // Parse input arrays from Python
    if (!PyArg_ParseTuple(args, "O!O!", 
                          &PyArray_Type, &x_array,
                          &PyArray_Type, &y_array)) {
        return NULL;
    }
    
    // Validate input dimensions
    if (PyArray_NDIM(x_array) != 2 || PyArray_NDIM(y_array) != 2) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be 2D");
        return NULL;
    }
    
    npy_intp* x_dims = PyArray_DIMS(x_array);
    npy_intp* y_dims = PyArray_DIMS(y_array);
    
    if (x_dims[0] != y_dims[0] || x_dims[1] != y_dims[1]) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must have same shape");
        return NULL;
    }
    
    npy_intp ny = x_dims[0];
    npy_intp nx = x_dims[1];
    
    // Create output array
    PyArrayObject* result = (PyArrayObject*)PyArray_SimpleNew(2, x_dims, NPY_FLOAT64);
    if (result == NULL) {
        return NULL;
    }
    
    // Get data pointers
    double* x_data = (double*)PyArray_DATA(x_array);
    double* y_data = (double*)PyArray_DATA(y_array);
    double* result_data = (double*)PyArray_DATA(result);
    
    // Process in batches for better cache locality
    const npy_intp batch_size = 256;
    for (npy_intp batch_start = 0; batch_start < ny; batch_start += batch_size) {
        npy_intp batch_end = (batch_start + batch_size < ny) ? batch_start + batch_size : ny;
        
        for (npy_intp i = batch_start; i < batch_end; i++) {
            for (npy_intp j = 0; j < nx; j++) {
                npy_intp idx = i * nx + j;
                result_data[idx] = calc_mandelbrot(x_data[idx], y_data[idx]);
            }
        }
    }
    
    return (PyObject*)result;
}

// Python module method definitions
static PyMethodDef MandelbrotMethods[] = {
    {"calc_val", calc_val, METH_VARARGS,
     "Compute Mandelbrot set values for meshgrid arrays"},
    {NULL, NULL, 0, NULL}
};

// Python module definition
static struct PyModuleDef mandelbrotmodule = {
    PyModuleDef_HEAD_INIT,
    "mandelbrot",
    "Fast Mandelbrot computation in C++",
    -1,
    MandelbrotMethods
};

// Module initialization function
PyMODINIT_FUNC PyInit_mandelbrot(void) {
    import_array();
    return PyModule_Create(&mandelbrotmodule);
}
