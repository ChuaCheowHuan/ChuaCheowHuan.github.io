---
layout: posts
author: Huan
title: Custom Sagemaker reinforcement learning container
---

Building & testing custom Sagemaker RL container.

Instead of using the official SageMaker supported version of Ray RLlib
(version 0.5.3 & 0.6.5), I want to use version 0.7.3. In order to do so, I have
to build & test my custom Sagemaker RL container.

---

**The Dockerfile:**

Add the Dockerfile below to ```sagemaker-rl-container/ray/docker/0.7.3/```:

```
ARG processor
#FROM 520713654638.dkr.ecr.us-west-2.amazonaws.com/sagemaker-tensorflow-scriptmode:1.14.0-$processor-py3
FROM 520713654638.dkr.ecr.us-west-2.amazonaws.com/sagemaker-tensorflow-scriptmode:1.12.0-$processor-py3

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        jq \
        libav-tools \
        libjpeg-dev \
        libxrender1 \
        python3.6-dev \
        python3-opengl \
        wget \
        xvfb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    Cython==0.29.7 \
    gym==0.14.0 \
    lz4==2.1.10 \
    opencv-python-headless==4.1.0.25 \
    PyOpenGL==3.1.0 \
    pyyaml==5.1.1 \
    redis>=3.2.2 \
    ray==0.7.3 \
    ray[rllib]==0.7.3 \
    scipy==1.3.0 \
    requests

# https://click.palletsprojects.com/en/7.x/python3/
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Copy workaround script for incorrect hostname
COPY lib/changehostname.c /

COPY lib/start.sh /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/start.sh

# Starts framework
ENTRYPOINT ["bash", "-m", "start.sh"]
```

---

**Remove unneeded test files:**

Backup the ```test``` folder as ```test_bkup```
in ```sagemaker-rl-container/```.

Remove the following files not used in testing
in ```sagemaker-rl-container/test/integration/local/```:

```
test_coach.py
test_vw_cb_explore.py
test_vw_cbify.py
test_vw_serving.py
```

---

**Add/replace codes in test files to get role:**

In the ```sagemaker-rl-container/test/conftest.py```file, add/replace the
following:

```
from sagemaker import get_execution_role
```

```
#parser.addoption('--role', default='SageMakerContainerBuildIntegrationTests')
parser.addoption('--role', default=get_execution_role()),

```

In the following files:

```
sagemaker-rl-container/test/integration/local/test_gym.py
sagemaker-rl-container/test/integration/local/test_ray.py
```

Add/replace the following:

```
from sagemaker import get_execution_role
```

```
#role='SageMakerRole',
role = get_execution_role(),
```

---

**Build the image:**

In SageMaker, start a Jupyter notebook instance & open a terminal.

Login into SageMaker ECR account:

```
$ (aws ecr get-login --no-include-email --region <region> --registry-ids <AWS_ACC_ID>)
$ (aws ecr get-login --no-include-email --region us-west-2 --registry-ids 520713654638)
```

Copy & paste the output from the above command into the terminal & press Enter.

---

Pull the base Tensorflow image from the aws ecr:

```
$ docker pull 520713654638.dkr.ecr.us-west-2.amazonaws.com/sagemaker-tensorflow-scriptmode:1.12.0-cpu-py3
```

Build the Ray image using the ```Dockerfile.tf``` from above:

```
$ docker build -t custom-smk-rl-ctn:tf-1.12.0-ray-0.7.3-cpu-py3 -f ray/docker/0.7.3/Dockerfile.tf --build-arg processor=cpu .
```

---

**Local testing:**

Install dependencies for testing:

```
$ cd sagemaker-rl-container
$ pip install .
```

Run the command below for local testing:

```
clear && \
docker images && \
pytest test/integration/local --framework tensorflow \
                              --toolkit ray \
                              --toolkit-version 0.7.3 \
                              --docker-base-name custom-smk-rl-ctn \
                              --tag tf-1.12.0-ray-0.7.3-cpu-py3 \
                              --processor cpu | tee local_test_output.txt
```

The output from the test will be saved in local_test_output.txt.

---

**Pushing to registry on AWS ECR:**

```
$ (aws ecr get-login --no-include-email --region <region> --registry-ids <AWS_ACC_ID>)
$ (aws ecr get-login --no-include-email --region us-west-2  --registry-ids 123456789012)
# Copy & paste output to terminal & press enter.

$ aws ecr create-repository --repository-name <repo_name>
$ aws ecr create-repository --repository-name custom-smk-rl-ctn

$ docker tag <image_ID> <AWS_ACC_ID>.dkr.ecr.us-west-2.amazonaws.com/<repo_name>:<tag>
$ docker tag ba542f0b9706 <123456789012>.dkr.ecr.us-west-2.amazonaws.com/custom-smk-rl-ctn:tf-1.12.0-cpu-py3
$ docker tag ba542f0b9706 <123456789012>.dkr.ecr.us-west-2.amazonaws.com/custom-smk-rl-ctn:tf-1.12.0-ray-0.7.3-cpu-py3

$ docker push <AWS_ACC_ID>.dkr.ecr.us-west-2.amazonaws.com/<repo_name>:<tag>
$ docker push <123456789012>.dkr.ecr.us-west-2.amazonaws.com/custom-smk-rl-ctn:tf-1.12.0-cpu-py3
$ docker push <123456789012>.dkr.ecr.us-west-2.amazonaws.com/custom-smk-rl-ctn:tf-1.12.0-ray-0.7.3-cpu-py3


$ aws ecr describe-repositories

$ aws ecr list-images --repository-name <repo_name>
$ aws ecr list-images --repository-name custom-smk-rl-ctn
```

---

**Testing with AWS SageMaker ML instance:**

Run the command below for testing with SageMaker:

```
clear && \
docker images && \
pytest test/integration/sagemaker --aws-id 123456789012 \
                                  --instance-type ml.m4.xlarge \
                                  --framework tensorflow \
                                  --toolkit ray \
                                  --toolkit-version 0.7.3 \
                                  --docker-base-name custom-smk-rl-ctn \
                                  --tag tf-1.12.0-ray-0.7.3-cpu-py3 | tee SageMaker_test_output.txt
```

The output from the test will be saved in SageMaker_test_output.txt.

---

**Pushing to registry on Docker hub:**

```
$ docker login

$ docker tag <image_ID> <DockerHubUserName>/<repo_name>:<tag>
$ docker tag ba542f0b9706 <DockerHubUserName>/custom-smk-rl-ctn:tf-1.12.0-cpu-py3
$ docker tag ba542f0b9706 <DockerHubUserName>/custom-smk-rl-ctn:tf-1.12.0-ray-0.7.3-cpu-py3

$ docker push <DockerHubUserName>/<repo_name>:<tag>
$ docker push <DockerHubUserName>/custom-smk-rl-ctn:tf-1.12.0-cpu-py3
$ docker push <DockerHubUserName>/custom-smk-rl-ctn:tf-1.12.0-ray-0.7.3-cpu-py3
```

---

**Training with custom SageMaker RL container:**



---

**Useful Docker commands:**

```
$ docker ps -a

$ docker images

$ docker rm <container>

$ docker rmi <image>
```

---

**Useful AWS commands:**

```
$ aws ecr delete-repository --force --repository-name <repo_name>
```

---

**References:**

[https://github.com/aws/sagemaker-rl-container](https://github.com/aws/sagemaker-rl-container)

---

<br>
