FROM pytorch:2.1.0-cu12.1.0-py3.8-ubuntu20.04
WORKDIR /etc
COPY requirement.txt .
RUN pip3 install -r requirement.txt 
RUN pip3 install einops