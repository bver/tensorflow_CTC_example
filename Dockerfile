
FROM nvidia/cuda:7.5-cudnn5-devel

# prerequisities
RUN apt-get update
RUN apt-get -y install git default-jdk wget libcurl4-gnutls-dev zlib1g-dev unzip swig python-dev python-numpy
WORKDIR /tf
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python get-pip.py
RUN pip install mock

# prepare bazel
RUN wget https://github.com/bazelbuild/bazel/releases/download/0.3.0/bazel-0.3.0-jdk7-installer-linux-x86_64.sh
RUN chmod a+x bazel-0.3.0-jdk7-installer-linux-x86_64.sh
RUN ./bazel-0.3.0-jdk7-installer-linux-x86_64.sh

# clone tensorflow
RUN git clone https://github.com/tensorflow/tensorflow.git

# apply tensorflow libcuda workaround using https://github.com/NVIDIA/nvidia-docker/issues/45
RUN ln -s /usr/local/nvidia/lib64/libcuda.so.1 /usr/local/cuda/lib64/libcuda.so
RUN ln -s /usr/local/nvidia/lib64/libcuda.so.1 /usr/local/cuda/lib64/libcuda.so.1

# build tensorflow
WORKDIR tensorflow
RUN /bin/bash -c 'echo -e "\nN\ny\n\n\n\n\n\n" | ./configure'
RUN ldconfig && bazel build -c opt --config=cuda //tensorflow/...

# install tensorflow
RUN bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/pip
RUN pip install --upgrade /tmp/pip/tensorflow-*.whl

