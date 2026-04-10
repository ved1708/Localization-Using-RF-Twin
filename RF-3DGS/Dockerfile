FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev build-essential \
    libgl1-mesa-glx libx11-6 libxext6 libxrender1 libxcb1 libxrandr2 libxi6 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip
RUN pip3 install open3d==0.17.0 plyfile numpy pillow matplotlib

WORKDIR /workspace
COPY . /workspace

# default is an interactive shell; examples below show how to run the viewer
ENTRYPOINT ["/bin/bash"]
