FROM nvcr.io/nvidia/tritonserver:23.04-py3

# COPY requirements.txt .
# RUN pip3 install -r requirements.txt
RUN apt-get update && apt-get install -y nvidia-container-toolkit
ENTRYPOINT [ "tritonserver" ]
