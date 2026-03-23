FROM vemlp-cn-beijing.cr.volces.com/preset-images/pytorch:2.1.0-cu12.1.0-py3.9-ubuntu20.04
ENV PIP_INDEX_URL=https://mirrors.ivolces.com/pypi/simple/
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV MMCV_WITH_OPS=1
ENV FORCE_CUDA=1
RUN pip install -U openmim
RUN pip install "opencv-python<=4.10.0.84"
RUN pip install scipy terminaltables pycocotools matplotlib yapf==0.32.0 numpy==1.24.4
RUN pip install "mmcv-full==1.7.2" -v --no-cache-dir
RUN mim install mmdet==2.28.2
RUN export MMCV_FILE=$(python -c "import site; print(site.getsitepackages()[0])")/mmcv/parallel/_functions.py && \
    echo "Patching file: $MMCV_FILE" && \
    sed -i "s/streams = \[_get_stream(device) for device in target_gpus\]/streams = \[_get_stream(torch.device('cuda', device)) for device in target_gpus\]/g" $MMCV_FILE

RUN export MMCV_DIST_PATH=$(python -c "import site; print(site.getsitepackages()[0])")/mmcv/parallel/distributed.py && \
    echo "Patching distributed.py..." && \
    sed -i "s/self._use_replicated_tensor_module/getattr(self, '_use_replicated_tensor_module', False)/g" $MMCV_DIST_PATH