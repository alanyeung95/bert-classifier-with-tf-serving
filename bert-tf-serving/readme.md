## Setup/Deployment
### Github Workflow Deployment
Please reference to: 
https://github.com/alanyeung95/bert-classifier-with-tf-serving/blob/main/.github/workflows/update-serving-model.yml

### Manual Deployment
Login into ec2 instance and run those cmd (my example)

```
# pull the tf-serving image
docker pull tensorflow/serving

# link model store in s3 bucket to instance storage
aws s3 sync s3://alanyeung-bert-classifier /home/ec2-user/model

docker run \
--name bert-tf-serving \
-p 8501:8501 \
--mount type=bind,source=/home/ec2-user/model/saved_model_bert,target=/models/bert \
--mount type=bind,source=/home/ec2-user/model/models.config,target=/models/models.config \
-e  MODEL_NAME=bert \
-t tensorflow/serving \
--model_config_file=/models/models.config 
```
