## Getting Started

Login into ec2 instance (for my own usage)
```
docker run -t --rm -d -p 8501:8501 -v "/home/ec2-user/model/saved_model_bert/:/models/bert" -e MODEL_NAME=bert  tensorflow/serving
```
