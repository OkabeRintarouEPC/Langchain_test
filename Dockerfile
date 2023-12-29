FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
# FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu18.04
# 设置时区
ENV TZ Asia/Shanghai
RUN apt-get update -y && \
    apt-get upgrade -y && \
    # apt-get install -y wget && \
    # 拒绝人工交互
    DEBIAN_FRONTEND="noninteractive" apt-get install -y vim python3-pip python3-venv libopencv-dev git-lfs sudo build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev libbz2-dev liblzma-dev && \
    # 清空apt缓存，并删除apt缓存文件
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
ADD . .
RUN pip install --upgrade pip && \
    pip config set global.index-url http://pypi.tuna.tsinghua.edu.cn/simple  && \
    pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn  && \
    pip install -r requirements.txt
    # python3 init_database.py --recreate-vs
EXPOSE 8501
EXPOSE 7861
EXPOSE 20000
EXPOSE 20002
ENTRYPOINT python3 init_database.py --recreate-vs && python3 startup.py -a
# CMD ["python3", "startup.py", "-a"]