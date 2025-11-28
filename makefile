PYTHON := python3
PYTHON_CONFIG := $(PYTHON)-config
PYTHON_INCLUDES := $(shell $(PYTHON_CONFIG) --includes)
PYTHON_LDFLAGS := $(shell $(PYTHON_CONFIG) --ldflags)
NUMPY_INCLUDE := $(shell $(PYTHON) -c "import numpy; print(numpy.get_include())")

CXX := g++
CXXFLAGS := -std=c++23 -O3 -fPIC -Wall $(PYTHON_INCLUDES) -I$(NUMPY_INCLUDE) -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
LDFLAGS := -shared $(PYTHON_LDFLAGS) -lpthread

EXT_SUFFIX := $(shell $(PYTHON_CONFIG) --extension-suffix)
TARGET_MANDELBROT := mandelbrot$(EXT_SUFFIX)
TARGET_IMAGE_PROCESSOR := image_processor$(EXT_SUFFIX)
SOURCE_MANDELBROT := mandelbrot.cpp
SOURCE_IMAGE_PROCESSOR := image_processor.cpp

.PHONY: all clean

all: $(TARGET_MANDELBROT) $(TARGET_IMAGE_PROCESSOR)

$(TARGET_MANDELBROT): $(SOURCE_MANDELBROT)
	$(CXX) $(CXXFLAGS) $(SOURCE_MANDELBROT) -o $(TARGET_MANDELBROT) $(LDFLAGS)

$(TARGET_IMAGE_PROCESSOR): $(SOURCE_IMAGE_PROCESSOR)
	$(CXX) $(CXXFLAGS) $(SOURCE_IMAGE_PROCESSOR) -o $(TARGET_IMAGE_PROCESSOR) $(LDFLAGS)

clean:
	rm -f $(TARGET_MANDELBROT) $(TARGET_IMAGE_PROCESSOR) output.fits mandelbrot_color.png temp_mandelbrot.raw mandelbrot_deepzoom.dzi
	rm -rf mandelbrot_deepzoom_files