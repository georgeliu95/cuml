#!/bin/bash
export CUML_BUILD_PATH=/rapids/cuml/cpp/build_devl
CUML_OLD_VERSION_DIR="/opt/conda/envs/rapids/lib/python3.8/site-packages/cuml"
rm -r ${CUML_OLD_VERSION_DIR}
python setup.py build_ext --inplace && python setup.py install


