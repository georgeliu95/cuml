#!/bin/bash
rm -r /rapids/cuml/cpp/examples/dbscan
ln -s /home/cpp/examples/dbscan /rapids/cuml/cpp/examples/dbscan

rm -r /rapids/cuml/cpp/src/dbscan
ln -s /home/cpp/src/dbscan /rapids/cuml/cpp/src/dbscan

rm /rapids/cuml/cpp/include/cuml/cluster/dbscan.hpp
ln -s /home/cpp/include/cuml/cluster/dbscan.hpp /rapids/cuml/cpp/include/cuml/cluster/dbscan.hpp
