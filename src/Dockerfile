FROM python:3.7-slim

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    gzip \
    libx11-6 \
    wget \
    unzip \
    cmake \
    libgtest-dev \
    libboost-test-dev \
    libboost-system-dev \
    libboost-thread-dev \
    libboost-program-options-dev \
    build-essential \
    libeigen3-dev \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
 && rm -rf /var/lib/apt/lists/*

# Install KENLM
RUN git clone https://github.com/kpu/kenlm && \
    cd kenlm && \
    python3 -m pip install . --install-option="--max_order 7" && \
    mkdir build && cd build && \
    cmake .. -DKENLM_MAX_ORDER=7 && \
    make -j all install

#Install MGIZA
RUN git clone https://github.com/moses-smt/mgiza.git && \
    cd /mgiza/mgizapp && \
    cmake . && \
    make && \
    make install && \
    cp /mgiza/mgizapp/scripts/merge_alignment.py /mgiza/mgizapp/bin
        
ENV PATH=$PATH:/mgiza/mgizapp/bin

#Install bicleaner
RUN git clone https://github.com/bitextor/bicleaner.git

#Install Moses
#RUN wget http://www.statmt.org/moses/RELEASE-4.0/binaries/ubuntu-17.04.tgz && tar zxf ubuntu-17.04.tgz && mv ubuntu-17.04 /mosesdecoder && rm ubuntu-17.04.tgz
RUN git clone https://github.com/SanderVanBeers/moses-smt-ubuntu1704-binaries.git && mv /moses-smt-ubuntu1704-binaries/* / && rm -rf moses-smt-ubuntu1704-binaries

#Create working environment
RUN mkdir /data
RUN mkdir /input
RUN mkdir /output
WORKDIR /workdir

COPY . /src

RUN pip install -r /src/requirements.txt

CMD ["python3" , "/src/main.py"]