#!/bin/bash
rm -r /rapids/cuml/cpp/examples/dbscan
ln -s /home/cpp/examples/dbscan /rapids/cuml/cpp/examples/dbscan

rm -r /rapids/cuml/cpp/src/dbscan
ln -s /home/cpp/src/dbscan /rapids/cuml/cpp/src/dbscan

rm /rapids/cuml/cpp/include/cuml/cluster/dbscan.hpp
ln -s /home/cpp/include/cuml/cluster/dbscan.hpp /rapids/cuml/cpp/include/cuml/cluster/dbscan.hpp

rm /rapids/cuml/cpp/src_prims/label/merge_labels.cuh
ln -s /home/cpp/src_prims/label/merge_labels.cuh /rapids/cuml/cpp/src_prims/label/merge_labels.cuh

rm /rapids/cuml/python/cuml/cluster/dbscan.pyx
ln -s /home/python/cuml/cluster/dbscan.pyx /rapids/cuml/python/cuml/cluster/dbscan.pyx

rm /rapids/cuml/cpp/build_devl/examples/dbscan/example.py
ln -s /home/cpp/examples/dbscan/example.py /rapids/cuml/cpp/build_devl/examples/dbscan/

ln -s /home/install_python.sh /rapids/cuml/python/install_python.sh
