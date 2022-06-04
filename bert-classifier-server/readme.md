## Getting Started

```
aws s3 sync s3://alanyeung-bert-classifier /home/ec2-user/model # link model store in s3 bucket to instance storage
cd bert-classifier-with-tf-serving/bert-classifier-server
docker build -t bert-classifier-server . &&  docker run -p 5000:5000 bert-classifier-server
```
