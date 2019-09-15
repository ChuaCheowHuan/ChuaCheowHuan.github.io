---
layout: posts
author: Huan
title: Custom Sagemaker reinforcement learning container
---

Infomation related to building custom Sagemaker RL container.

---

**Docker commands:**

```
$ docker login

$ docker ps -a

$ docker images

$ docker tag ba542f0b9706 123456789012.dkr.ecr.us-west-2.amazonaws.com/custom-smk-rl-ctn:tf-1.12.0-cpu-py3

$ docker push 123456789012.dkr.ecr.us-west-2.amazonaws.com/custom-smk-rl-ctn:tf-1.12.0-cpu-py3

$ docker pull 123456789012.dkr.ecr.us-west-2.amazonaws.com/custom-smk-rl-ctn:tf-1.12.0-cpu-py3
```

---

**AWS commands:**

```
$ (aws ecr get-login --no-include-email --region us-west-2 --registry-ids 123456789012)
[copy & paste, enter]

$ aws ecr describe-repositories

$ aws ecr list-images --repository-name custom-smk-rl-ctn
```

---

**Sagemaker testing commands:**

```
# local test:
$ pytest test/integration/local --toolkit ray \
                                --docker-base-name 123456789012.dkr.ecr.us-west-2.amazonaws.com/custom-smk-rl-ctn \
                                --tag ray-0.7.3-cpu-py3 \
                                --processor cpu

# SageMaker test:
$ pytest test/integration/sagemaker --toolkit ray \
                                    --aws-id 123456789012 \
                                    --docker-base-name custom-smk-rl-ctn \
                                    --instance-type ml.m4.xlarge \
                                    --tag ray-0.7.3-cpu-py3
```

---

[Reference](https://github.com/aws/sagemaker-rl-container)

---

<br>
