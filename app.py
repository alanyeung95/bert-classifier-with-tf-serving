import os
import sys
import logging

import requests
import tensorflow as tf
from official.nlp import bert
import official.nlp.bert.tokenization
import tensorflow_datasets as tfds
import numpy as np
from flask import Flask, jsonify, request

# Load the required submodules
import official.nlp.bert.bert_models
import official.nlp.bert.configs

import logging
import coloredlogs
import verboselogs
verboselogs.install()
coloredlogs.install(
    fmt = "[%(asctime)s.%(msecs)03d][%(levelname)s][%(name)s][%(module)s:%(lineno)d] %(message)s",
    field_styles = {'asctime': {'color': 'green'}, 'msecs': {'color': 'green'}, 'hostname': {'color': 'magenta'}, 'levelname': {'bold': True, 'color': 'black'}, 'name': {'color': 'blue'}, 'programname': {'color': 'cyan'}, 'username': {'color': 'yellow'}},
    datefmt = "%Y-%m-%d %H:%M:%S",
    stream = sys.stdout,
    level = 15,
)

LOG_LEVEL = int(os.getenv("LOG_LEVEL", 20))

# logger config
logger = logging.getLogger()
logger.setLevel(LOG_LEVEL)
logging.getLogger('werkzeug').disabled = True

gs_folder_bert = "gs://cloud-tpu-checkpoints/bert/v3/uncased_L-12_H-768_A-12"


# GLUE, the General Language Understanding Evaluation benchmark (https://gluebenchmark.com/) 
# is a collection of resources for training, evaluating, and analyzing natural language understanding systems.

glue, info = tfds.load('glue/mrpc', with_info=True,
                       # It's small, load the whole dataset
                       batch_size=-1)

glue_train = glue['train']

tokenizer = bert.tokenization.FullTokenizer(
    vocab_file=os.path.join(gs_folder_bert, "vocab.txt"),
     do_lower_case=True)

tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])

print("Vocab size:", len(tokenizer.vocab))

def encode_sentence(s, tokenizer):
   tokens = list(tokenizer.tokenize(s))
   tokens.append('[SEP]')
   return tokenizer.convert_tokens_to_ids(tokens)

def bert_encode(glue_dict, tokenizer):
  num_examples = len(glue_dict["sentence1"])
  
  sentence1 = tf.ragged.constant([
      encode_sentence(s, tokenizer)
      for s in np.array(glue_dict["sentence1"])])
  sentence2 = tf.ragged.constant([
      encode_sentence(s, tokenizer)
       for s in np.array(glue_dict["sentence2"])])

  cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])]*sentence1.shape[0]
  input_word_ids = tf.concat([cls, sentence1, sentence2], axis=-1)

  input_mask = tf.ones_like(input_word_ids).to_tensor()

  type_cls = tf.zeros_like(cls)
  type_s1 = tf.zeros_like(sentence1)
  type_s2 = tf.ones_like(sentence2)
  input_type_ids = tf.concat(
      [type_cls, type_s1, type_s2], axis=-1).to_tensor()

  inputs = {
      'input_word_ids': input_word_ids.to_tensor(),
      'input_mask': input_mask,
      'input_type_ids': input_type_ids}

  return inputs

my_examples = bert_encode(
    glue_dict = {
        'sentence1':[
            'The rain in Spain falls mainly on the plain.',
            'Look I fine tuned BERT.'],
        'sentence2':[
            'It mostly rains on the flat lands of Spain.',
            'Is it working? This does not match.']
    },
    tokenizer=tokenizer)

import json

bert_config_file = os.path.join(gs_folder_bert, "bert_config.json")
config_dict = json.loads(tf.io.gfile.GFile(bert_config_file).read())

bert_config = bert.configs.BertConfig.from_dict(config_dict)

_, bert_encoder = bert.bert_models.classifier_model(
    bert_config, num_labels=2)



app = Flask(__name__)

@app.route("/")
def home():
    return "Hello, World!"

@app.route("/predict", methods=["POST"])
def predict():
    text = request.json["glue_dict"]

    encoded = bert_encode(text, tokenizer=tokenizer)
    inputJson = {
        "input_word_ids": encoded["input_word_ids"].numpy().tolist(),
        "input_mask": encoded["input_mask"].numpy().tolist(),
        "input_type_ids": encoded["input_type_ids"].numpy().tolist()
    }

    response = requests.post(os.environ["BERT_CLASSIFIER_HOST"] + ":8501/v1/models/bert" + ":predict", json={"inputs": inputJson}, timeout=2)
    response_json = response.json()
    result = tf.argmax(response_json["outputs"],1).numpy()

    return {"result": result.tolist()}

if __name__ == "__main__":
    app.run(debug=True)
