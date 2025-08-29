# hash:sha256:0130dbbccf3c3b210bc3b99d61236081618ff7db117171c6bce1ffc8453e5676
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    dynamax[notebooks]==1.0.1 \
    hdmf==3.14.6 \
    hdmf-zarr==0.11.3 \
    matplotlib==3.10.5 \
    numpy==2.2.6 \
    nwbwidgets==0.11.3 \
    pandas==2.3.2 \
    pynwb==3.0.0 \
    scikit-learn==1.7.1 \
    umap-learn==0.5.9.post2
