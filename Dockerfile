FROM python:3.8.9

WORKDIR /bert

ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000
ENV BERT_CLASSIFIER_HOST=ec2-44-193-17-101.compute-1.amazonaws.com

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "flask", "run" ]
