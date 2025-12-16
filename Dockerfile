# Use an official Miniconda3 image as a base
FROM continuumio/miniconda3

# Install OpenSSH server
RUN apt-get update && apt-get install -y openssh-server && \
    mkdir /var/run/sshd
    
WORKDIR /app

# 3. 아까 만든 환경 목록(yaml)을 도커 안으로 복사
COPY environment_jaewon.yaml .

# 4. 목록을 보고 Conda 환경 생성
RUN conda env create -f environment_jaewon.yaml

# 5. (중요) 도커가 켜질 때 자동으로 이 환경이 켜지도록 설정
# 아래 [본인환경이름]을 environment.yaml 파일 안의 name: 옆에 있는 이름으로 바꿔주세요!
RUN echo "conda activate dit4sr" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]