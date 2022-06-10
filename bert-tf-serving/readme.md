## Getting Started

Login into ec2 instance (for my own usage)
```
docker run \
--name bert-tf-serving \
-p 8501:8501 \
--mount type=bind,source=/home/ec2-user/model/saved_model_bert,target=/models/bert \
--mount type=bind,source=/home/ec2-user/model/models.config,target=/models/models.config \
-e  MODEL_NAME=bert \
-t tensorflow/serving \
--model_config_file=/models/models.config 
```
