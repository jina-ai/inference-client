# Inference Client

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

### Client Initialization

To initialize the Client object and connect to the inference server, you can choose to pass a valid personal access token to the token parameter.
A personal access token can be generated at the [Jina AI Cloud](https://cloud.jina.ai/settings/tokens), or via CLI as described in [this guide](https://docs.jina.ai/jina-ai-cloud/login/#create-a-new-pat):
```bash
jina auth token create <name of PAT> -e <expiration days>
```

To pass the token to the client, you can use the following code snippet:

```python
from client import Client

# Initialize client with valid token
client = Client(token='valid token')
```

If you don't provide a token explicitly, Inference Client will look for a `JINA_AUTH_TOKEN` environment variable, otherwise it will try to authenticate via browser.

```python
from client import Client

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
from client import Client

# Initialize client
inference_client = Client()

# Connect to CLIP model
clip_model = inference_client.get_model('ViT-B-32::openai')
clip_embed = clip_model.encode(text='hello world')[0].embeddings

# Connect to BLIP2 model
blip2_model = inference_client.get_model('Salesforce/blip2-opt-2.7b')
blip2_embed = blip2_model.encode(text='hello jina')[0].embeddings
```

Now it's time to use the models to perform some tasks.
We will use the Singapore Skyline with Merlion in the foreground as an example image for the rest of the examples.

<p align="center">
    <img src=".github/README-img/Singapore_Skyline_2019-10.jpeg" width="50%">
</p>

### Encoding

To use the encode method of an inference model, you need to initialize the model and provide input data as DocumentArray, plain text, or an image. 

Here are some examples of how to use the encode method:

1. Encode a `DocumentArray`:

```python
from client import Client
from docarray import Document, DocumentArray

# Initialize client
inference_client = Client()

# Connect to CLIP model
clip_model = inference_client.get_model('ViT-B-32::openai')

# Create a DocumentArray with two documents
docs = DocumentArray([Document(text='hello world'), Document(uri='singapore.jpg')])

# Encode the documents
response = clip_model.encode(docs=docs)

# Access the embeddings
for doc in response:
    print(doc.embedding)
```

2. Encode plain text:

```python
# Encode the documents
response = clip_model.encode(text='hello world')

# Access the embeddings
print(response[0].embedding)
```

3. Encode an image:

```python
# Encode image URL
response = clip_model.encode(image='singapore.jpg')

# Access the embedding
print(response[0].embedding)

# Encode image binary data
image_bytes = open('my_image.jpg', 'rb').read()
response = clip_model.encode(image=image_bytes)

# Access the embedding
print(response[0].embedding)

# Encode image tensor data
import torch
from PIL import Image
import torchvision.transforms as transforms

image_bytes = Image.open('my_image.jpg')
transform = transforms.ToTensor()
image_tensor = transform(image_bytes)
response = clip_model.encode(image=image_tensor)

# Access the embedding
print(response[0].embedding)
```

### Ranking

To perform similarity-based ranking of candidate matches, you can use the rank method of an inference model. 
The rank method takes a reference input and a list of candidates, and reorder that list of candidates based on their similarity to the reference input. 
You can also construct a cross-modal Document where the root contains an image or text and .matches contain images or sentences to rerank.

Here are some examples of how to use the rank method:

1. Rank a 'DocumentArray':

```python
from client import Client
from docarray import Document, DocumentArray

# Initialize client
inference_client = Client()

# Initialize model
clip_model = Client().get_model('ViT-B-32::openai')

# Create a DocumentArray with a single document and some candidate matches
docs = DocumentArray(
    [
        Document(
            uri='singapore.jpg',
            matches=DocumentArray(
                [
                    Document(text='a colorful photo of nature'),
                    Document(text='a black and white photo of a dog'),
                    Document(text='a black and white photo of a cat'),
                ]
            ),
        ),
    ]
)

# Rank the documents
response = clip_model.rank(docs=docs)

# Access the matches
for doc in response:
    print(doc.matches)
```

2. Rank plain input:

```python
reference = 'singapore.jpg'
candidates = [
    'a colorful photo of nature',
    'a black and white photo of a dog',
    'a black and white photo of a cat',
]
response = clip_model.rank(reference=reference, candidates=candidates)

# Access the matches
print(response[0].matches)
```

**NOTICE**: The following tasks Caption and VQA are BLIP2 exclusive. Calling these methods on other models will fall back to the default encode method.

### Captioning

You can use caption to generate natural language descriptions of images.
The caption method takes a DocumentArray containing images or a single plain image as input.
The plain input image can be in the form of a URL string, an image blob, or an image tensor.

Here are some examples of how to use the caption method:

1. Caption a 'DocumentArray':

```python
from client import Client
from docarray import Document, DocumentArray

# Initialize client
inference_client = Client()

# Initialize model
blip2_model = Client().get_model('Salesforce/blip2-opt-2.7b')

# Create a DocumentArray with a single image document
docs = DocumentArray([Document(uri='singapore.jpg')])

# Rank the documents
response = blip2_model.caption(docs=docs)

# Access the matches
for doc in response:
    print(doc.tags['response'])
```

2. Caption plain input:

```python
```

## Support

- Join our [Slack community](https://slack.jina.ai) and chat with other community members about ideas.
- Watch our [Engineering All Hands](https://youtube.com/playlist?list=PL3UBBWOUVhFYRUa_gpYYKBqEAkO4sxmne) to learn Jina's new features and stay up-to-date with the latest AI techniques.
- Subscribe to the latest video tutorials on our [YouTube channel](https://youtube.com/c/jina-ai)

## License

Inference Client is backed by [Jina AI](https://jina.ai) and licensed under [Apache-2.0](./LICENSE). 