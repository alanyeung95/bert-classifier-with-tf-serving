# bert-classifier-with-tf-serving
It is a classifier used for determining whether the sentences in the pair are semantically equivalent.

# Usage
Let's use curl cmd to try the API
```
curl --location --request POST 'ec2-44-193-17-101.compute-1.amazonaws.com:5000/predict' \
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
