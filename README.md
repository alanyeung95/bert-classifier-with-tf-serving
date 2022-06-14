# bert-classifier-with-tf-serving
It is a classifier used for determining whether the sentences in the pair are semantically equivalent.

# Getting Started

## Cloud instance setting

After renting two machines in aws ec2, use the following commands to initial both instances (bert-classifier-server and bert-tf-serving )
```
sudo yum update -y 
sudo yum install git -y
sudo yum install gcc -y
sudo yum install docker -y
sudo service docker start
sudo usermod -a -G docker ec2-user # need to restart session to refresh the changes
```

Then please refer to the readme insider bert-classifier-server and bert-tf-serving folder to continue the setting

# Usage
Let's use curl cmd to try the API
```
curl --location --request POST 'ec2-44-193-17-101.compute-1.amazonaws.com:5000/predict/{model_version}' \
--header 'Content-Type: application/json' \
--data-raw '{
    "glue_dict": {
        "sentence1": [
            "The rain in Spain falls mainly on the plain.",
            "Look I fine tuned BERT.",
            "i love you",
            "i love you",
            "i love you"
        ],
        "sentence2": [
            "It mostly rains on the flat lands of Spain.",
            "Is it working? This does not match.",
            "i love you too",
            "i hate you",
            "they are not the same"
        ]
    }
}'
```

sample response:
```
{
    "result": [
        1,
        0,
        1,
        0,
        0
    ]
}
```
