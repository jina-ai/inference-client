# Inference Client

[![PyPI](https://img.shields.io/pypi/v/inference-client)](https://pypi.org/project/inference-client/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/inference-client)](https://pypi.org/project/inference-client/)
[![PyPI - License](https://img.shields.io/pypi/l/inference-client)](https://pypi.org/project/inference-client/)

Inference Client is a library that provides a simple and efficient way to use Jina AI's Inference, a powerful platform that offers a range of AI models for common tasks such as visual reasoning, question answering, and embedding modalities like texts and images. 
With Inference Client, you can easily select the task and model of your choice and integrate the API call into your workflow with zero technical overhead. 

The current version of Inference Client includes methods to call the following tasks:

📈 **Encode**: Encode data into embeddings using various models 

🔍 **Rank**: Re-rank cross-modal matches according to their joint likelihood

📷 **Caption**: Generate captions for images 

🤔 **VQA**: Answer questions related to images 

In addition to these tasks, the client provides the ability to connect to the inference server and user authentication.

## Installation

Please note that Inference Client requires Python 3.8 or higher. Inference Client can be installed via pip by executing:
```bash
pip install inference-client
```

## Getting Started

Before using Inference Client, please create an inference on [Jina AI Cloud](https://cloud.jina.ai/user/inference).

After login with your Jina AI Cloud account, you can create an inference by clicking the "Create" button in the inference page.
From there, you can select the model you want to use.

After the inference is created and the status is "Serving", you can use Inference Client to connect to it.
This could take a few minutes, depending on the model you selected.

<p align="center">
    <img src=".github/README-img/jac.png">
</p>

### Client Initialization

To initialize the Client object and connect to the inference server, you can choose to pass a valid personal access token to the token parameter.
A personal access token can be generated at the [Jina AI Cloud](https://cloud.jina.ai/settings/tokens), or via CLI as described in [this guide](https://docs.jina.ai/jina-ai-cloud/login/#create-a-new-pat):
```bash
jina auth token create <name of PAT> -e <expiration days>
```

To pass the token to the client, you can use the following code snippet:

```python
from inference_client import Client

# Initialize client with valid token
client = Client(token='<your token>')
```

If you don't provide a token explicitly, Inference Client will look for a `JINA_AUTH_TOKEN` environment variable, otherwise it will try to authenticate via browser.

```python
from inference_client import Client

client = Client()
```

Please note that while it's possible to login via the Jina AI web UI, this method is intended primarily for development and testing purposes. 
For production use, we recommend obtaining a long-lived token via the Jina AI web API and providing it to the Client object explicitly. 
Tokens have a longer lifetime than web sessions and can be securely stored and managed, making them a more suitable choice for production environments.

### Selecting the Model

To select an inference model, you can use the get_model method of the Client object and specify the name of the model as it appears in Jina AI Cloud. 
You can connect to as many inference models as you want once they have been created on Jina AI Cloud, and you can use them for multiple tasks.

Here's an example of how to connect to two models and encode some text using each of them:
    
```python
from inference_client import Client

# Initialize client
inference_client = Client()

# Connect to CLIP model
clip_model = inference_client.get_model('ViT-B-32::openai')
clip_embed = clip_model.encode(text='hello world')[0].embedding

# Connect to BLIP2 model
blip2_model = inference_client.get_model('Salesforce/blip2-opt-2.7b')
blip2_embed = blip2_model.encode(text='hello jina')[0].embedding
```

Now it's time to use the models to perform some tasks!
We will use the Singapore Skyline with Merlion in the foreground as an example image for the rest of the examples.

<p align="center">
    <img src=".github/README-img/Singapore_Skyline_2019-10.jpeg" width="50%">
</p>

### Encoding

To use the encode method of an inference model, you need to initialize the model and provide input data as DocumentArray, plain text, or an image. 

Here are some examples of how to use the encode method:

1. Encode plain text:

```python
from inference_client import Client

# Initialize client
inference_client = Client()

# Connect to CLIP model
model = inference_client.get_model('<inference model name>')

# Encode the documents
response = model.encode(text='hello world')

# Access the embeddings
print(response[0].embedding)
```

```bash
[-5.48706055e-02 -1.10717773e-01  5.13671875e-01 -3.22509766e-01
 -1.40380859e-01  6.23535156e-01  3.07617188e-01  4.26025391e-01
  ...
  8.04443359e-02  8.53515625e-01 -5.96008301e-02  3.61633301e-02]
```

2. Encode an image:

```python
# Encode image URL
response = model.encode(image='singapore.jpg')

# Access the embedding
print(response[0].embedding)

# Encode image binary data
image_bytes = open('singapore.jpg', 'rb').read()
response = model.encode(image=image_bytes)

# Access the embedding
print(response[0].embedding)

# Encode image tensor data
from PIL import Image
from numpy import asarray

image_bytes = Image.open('singapore.jpg')
image_tensor = asarray(image_bytes)
response = model.encode(image=image_tensor)

# Access the embedding
print(response[0].embedding)
```

```bash

[-1.70776367e-01 -4.17236328e-01  2.29370117e-01  1.95770264e-02
 -5.86914062e-01  1.30981445e-01 -2.38037109e-01 -1.24328613e-01
  ...
  2.59277344e-01  7.36694336e-02  4.23339844e-01 -2.92480469e-01]
```

3. Encode a `DocumentArray`:

```python
from docarray import Document, DocumentArray

# Create a DocumentArray with two documents
docs = DocumentArray([Document(text='hello world'), Document(uri='singapore.jpg')])

# Encode the documents
response = model.encode(docs=docs)

# Access the embeddings
for doc in response:
    print(doc.embedding)
```

```bash
[-5.48706055e-02 -1.10717773e-01  5.13671875e-01 -3.22509766e-01
 -1.40380859e-01  6.23535156e-01  3.07617188e-01  4.26025391e-01
  ...
  8.04443359e-02  8.53515625e-01 -5.96008301e-02  3.61633301e-02]
[ 1.26416489e-01  2.53842145e-01  1.32031530e-01 -6.55740649e-02
  3.77700478e-01  1.34678692e-01  1.94542333e-01  6.93580136e-04
  ...
  1.24198742e-01  2.51199156e-02 -1.18231498e-01  1.66848406e-01]
```

### Ranking

To perform similarity-based ranking of candidate matches, you can use the rank method of an inference model. 
The rank method takes a reference input and a list of candidates, and reorder that list of candidates based on their similarity to the reference input. 
You can also construct a cross-modal Document where the root contains an image or text and .matches contain images or sentences to rerank.

Here are some examples of how to use the rank method:

1. Rank plain input:

```python
from inference_client import Client

# Initialize client
inference_client = Client()

# Initialize model
model = Client().get_model('<inference model name>')

reference = 'singapore.jpg'
candidates = [
    'a colorful photo of nature',
    'a photo of blue scenery',
    'a black and white photo of a cat',
]
response = model.rank(reference=reference, candidates=candidates)

# Access the matches
for match in not response[0]:
    print(match.text)
```

```bash
a photo of blue scenery
a colorful photo of nature
a black and white photo of a cat
```
You may also input images as bytes or tensors similarly to the encode method.

2. Rank a `DocumentArray`:

```python
from docarray import Document, DocumentArray

# Create a DocumentArray with a single document and some candidate matches
docs = DocumentArray(
    [
        Document(
            uri='singapore.jpg',
            matches=DocumentArray(
                [
                    Document(text='a colorful photo of nature'),
                    Document(text='a photo of blue scenery'),
                    Document(text='a black and white photo of a cat'),
                ]
            ),
        ),
    ]
)

# Rank the documents
response = model.rank(docs=docs)

# Access the matches
for match in not response[0]:
    print(match.text)
```

```bash
a photo of blue scenery
a colorful photo of nature
a black and white photo of a cat
```

**NOTICE**: The following tasks Caption and VQA are BLIP2 exclusive. Calling these methods on other models will fall back to the default encode method.

### Captioning

You can use caption to generate natural language descriptions of images.
The caption method takes a DocumentArray containing images or a single plain image as input.
The plain input image can be in the form of a URL string, an image blob, or an image tensor.

Here are some examples of how to use the caption method:

1. Caption plain input:

```python
from inference_client import Client

# Initialize client
inference_client = Client()

# Initialize model
model = Client().get_model('<inference model name>')

response = model.caption(image='singapore.jpg')

# Access the captions
print(response[0].tags['response'])
```

```bash
the merlion fountain in singapore at night
```

You may also input images as bytes or tensors similarly to the encode method.

2. Caption a `DocumentArray`:

```python
from docarray import Document, DocumentArray

# Create a DocumentArray with a single image document
docs = DocumentArray([Document(uri='singapore.jpg')])

# Caption the documents
response = model.caption(docs=docs)

# Access the captions
for doc in response:
    print(doc.tags['response'])
```

```bash
the merlion fountain in singapore at night
```

### Visual Question Answering

Visual Question Answering (VQA) is a task that involves answering natural language questions about visual content such as images. 
Given an image and a question, the goal of VQA is to provide a natural language answer.
The VQA method takes either a DocumentArray of images and questions, or a single plain image and question.

Here are some examples of how to use the VQA method:

2. VQA plain input:

```python
from inference_client import Client

# Initialize client
inference_client = Client()

# Initialize model
model = Client().get_model('<inference model name>')

image = 'singapore.jpg'
question = 'Question: What is this photo about? Answer:'

response = model.vqa(image=image, question=question)

# Access the answers
print(response[0].tags['response'])
```

```bash
the merlion fountain in singapore
```

You may also input images as bytes or tensors similarly to the encode method.
Please notice that due to the limitation of the current model, the question must start with 'Question:' and end with 'Answer:'.

2. VQA a `DocumentArray`:

```python
from docarray import Document, DocumentArray

# Create a DocumentArray with one document
docs = DocumentArray(
    [
        Document(
            uri='singapore.jpg',
            tags={'prompt': 'Question: What is this photo about? Answer:'},
        )
    ]
)

# VQA the documents
response = model.vqa(docs=docs)

# Access the answers
for doc in response:
    print(doc.tags['response'])
```

```bash
the merlion fountain in singapore
```

## Support

- Join our [Slack community](https://slack.jina.ai) and chat with other community members about ideas.
- Watch our [Engineering All Hands](https://youtube.com/playlist?list=PL3UBBWOUVhFYRUa_gpYYKBqEAkO4sxmne) to learn Jina's new features and stay up-to-date with the latest AI techniques.
- Subscribe to the latest video tutorials on our [YouTube channel](https://youtube.com/c/jina-ai)

## License

Inference Client is backed by [Jina AI](https://jina.ai) and licensed under [Apache-2.0](./LICENSE). 