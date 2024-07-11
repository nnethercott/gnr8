FROM python:3.10-slim-bookworm

RUN apt update -y && apt upgrade -y 

RUN apt-get install -y \
    build-essential \
    curl \
    vim

# Get Rust
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
RUN echo 'source $HOME/.cargo/env' >> $HOME/.bashrc

# install torch 
RUN pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu118
RUN pip install numpy==1.26

CMD ["python"]
