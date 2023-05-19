<p align="center">
<br>
<a href="https://cloud.jina.ai/user/inference"><img src="https://github.com/jina-ai/inference-client/blob/main/.github/README-img/inference_client.svg?raw=true" alt="" width="360px"></a>
<br>
</p>

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


## Installation

Inference Client is available on PyPI and can be installed using pip:

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
    <img src=".github/README-img/jac.png" width="100%">
</p>

### Client Initialization

To use Inference Client, you need to initialize a Client object with the authentication token of your Jina AI Cloud account:

```python
from inference_client import Client

client = Client(token='<your auth token>')
```

The token can be generated at the [Jina AI Cloud](https://cloud.jina.ai/settings/tokens), or via CLI as described in [this guide](https://docs.jina.ai/jina-ai-cloud/login/#create-a-new-pat):
```bash
jina auth token create <name of PAT> -e <expiration days>
```


### Connecting to Models

Once you have initialized the Client object, you can connect to the models you want to use by calling the `get_model` method, which takes the name of the model as it appears in Jina AI Cloud as an argument.

```python
# connect to a CLIP model
model = client.get_model('ViT-B-32::openai')
```
As example, the above code connects to the CLIP model named "ViT-B-32::openai" on Jina AI Cloud.

You can connect to as many inference models as you want once they have been created on Jina AI Cloud, and you can use them for multiple tasks.


### Performing tasks

Now that you have connected to the models, you can use them to perform the tasks they support.


#### 1. Encoding

The encode task is used to encode data into embeddings using various models.
For example, you can use the CLIP model to encode text or images into embeddings:
    
```python
model = client.get_model(
    '<name of the model that supports encode>'
)  # e.g. ViT-B-32::openai

# encode text
result = model.encode(text='hello world')

# encode image
result = model.encode(image='hello_world.jpg')

# encode image RGB tensor
from PIL import Image
from numpy import asarray

image_bytes = Image.open('hello_world.jpg')
image_tensor = asarray(image_bytes)
result = model.encode(image=image_tensor)
```

The output of the encode method is a DocumentArray, which contains the embeddings of the input data.

```bash
# print(result[0].embedding)

[-5.48706055e-02 -1.10717773e-01  5.13671875e-01 -3.22509766e-01
 -1.40380859e-01  6.23535156e-01  3.07617188e-01  4.26025391e-01
  ...
  8.04443359e-02  8.53515625e-01 -5.96008301e-02  3.61633301e-02]
```

### 2. Ranking

To perform similarity-based ranking of candidate matches, you can use the `rank` method of an inference model. 
The rank method takes a reference input and a list of candidates, and reorder that list of candidates based on their similarity to the reference input. 
You can also construct a cross-modal Document where the root contains an image or text and `.matches` contain images or sentences to rerank.

```python
# Connect to a model
model = client.get_model('<name of the model that supports rank>')

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

**NOTICE**: The following tasks Caption and VQA are BLIP2 exclusive. Calling these methods on other models will fall back to the default encode method.

### 3. Captioning

You can use caption method to generate natural language descriptions of images.


The caption method takes a DocumentArray containing images or a single plain image as input.
The plain input image can be in the form of a URL string, an image blob, or an image tensor.

For example, you can use the BLIP2 model to generate captions for images:

```python
# Initialize model
model = client.get_model('Salesforce/blip2-opt-2.7b')

response = model.caption(image='singapore.jpg')

# Access the captions
print(response[0].tags['response'])
```

```bash
the merlion fountain in singapore at night
```


### 4. VQA (Visual Question Answering)

Visual Question Answering (VQA) is a task that involves answering natural language questions about visual content such as images. 
Given an image and a question, the goal of VQA is to provide a natural language answer.
The VQA method takes either a DocumentArray of images and questions, or a single plain image and question.


```python
# Initialize model
model = client.get_model('Salesforce/blip2-opt-2.7b')

image = 'singapore.jpg'
question = 'Question: What is this photo about? Answer:'

response = model.vqa(image=image, question=question)

# Access the answers
print(response[0].tags['response'])
```

```bash
the merlion fountain in singapore
```

## Documentation

For more information about advanced usage of Inference Client, please refer to the [documentation](https://jina.readme.io/docs/inference).

## Support

- Join our [Slack community](https://slack.jina.ai) and chat with other community members about ideas.
- Watch our [Engineering All Hands](https://youtube.com/playlist?list=PL3UBBWOUVhFYRUa_gpYYKBqEAkO4sxmne) to learn Jina's new features and stay up-to-date with the latest AI techniques.
- Subscribe to the latest video tutorials on our [YouTube channel](https://youtube.com/c/jina-ai)

## License

Inference Client is backed by [Jina AI](https://jina.ai) and licensed under [Apache-2.0](./LICENSE). 