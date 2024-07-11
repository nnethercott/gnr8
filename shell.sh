#!/bin/bash 

docker run -it -v $(pwd):/app --runtime=nvidia --gpus all gnr8:latest /bin/bash
