FROM python:3.10-slim-bookworm as build 

RUN apt update -y && apt upgrade -y 

RUN apt-get install -y \
    build-essential \
    curl \
    vim \
    git

# Get Rust
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y && \
    echo 'source $HOME/.cargo/env' >> $HOME/.bashrc

# Install torch and numpy
RUN pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu118

# Clone the repository and build the project
RUN git clone https://github.com/nnethercott/gnr8.git && \
    cd gnr8 && \
    /root/.cargo/bin/cargo build && \
    mkdir /app && cp target/debug/libgnr8.so /app/gnr8.so


# second stage
FROM python:3.10-slim-bookworm

WORKDIR /app
COPY --from=build /app/gnr8.so /app/gnr8.so

RUN pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cu118 && \
    pip install --no-cache-dir numpy==1.26    

CMD ["python"]
