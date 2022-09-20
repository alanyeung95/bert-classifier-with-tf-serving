## Overview

This python flask server will receive API request (sentences) and send a prediction request to tf-serving, finally will return the result to end-user

## Deployment

### Github workflow deployment

You can take a reference on how I config the github workflow deployment:
https://github.com/alanyeung95/bert-classifier-with-tf-serving/blob/main/.github/workflows/docker-image.yml

### Manual deployment
or if you have your own instance, you can also setup the flask server manually on instance

```
# please try to pull your server source code
cd bert-classifier-with-tf-serving/bert-classifier-server
# build the docker image and run it
docker build -t bert-classifier-server . &&  docker run -p 5000:5000 bert-classifier-server
```
