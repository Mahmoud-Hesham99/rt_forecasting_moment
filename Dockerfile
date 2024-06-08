FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04 as builder

# Update and install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    dos2unix \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Add deadsnakes PPA for newer Python versions
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-distutils \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3

RUN apt-get update && apt-get install -y --no-install-recommends \
curl \
&& rm -rf /var/lib/apt/lists/* \
&& curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
&& python3.10 get-pip.py \
&& rm get-pip.py

# Copy the Python dependencies file and install dependencies
COPY ./requirements.txt .
RUN python3.10 -m pip install -r requirements.txt

COPY src ./opt/src

COPY ./entry_point.sh /opt/
RUN chmod +x /opt/entry_point.sh

WORKDIR /opt/src

ENV MPLCONFIGDIR=/tmp/matplotlib
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/app:${PATH}"


RUN mkdir -p /opt/src/.cache/huggingface && chmod -R 777 /opt/src/.cache
ENV HF_HOME=/opt/src/.cache/huggingface

# set non-root user
USER 1000
# set entrypoint
ENTRYPOINT ["/opt/entry_point.sh"]
