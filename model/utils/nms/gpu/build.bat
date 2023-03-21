del *.lib
del *.cpp
rmdir /s /q build

nvcc -lib -O3 -arch=sm_86 -Xptxas="-v" -o nms_kernel.lib gpu_nms.cu

python build.py build_ext -i

del *.lib
del *.cpp
rmdir /s /q build