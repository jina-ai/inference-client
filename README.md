<p align="center">
<br>
<a href="https://cloud.jina.ai/user/inference"><img src="https://github.com/jina-ai/inference-client/blob/main/.github/README-img/inference_client.svg?raw=true" alt="" width="360px"></a>
<br>
</p>

[![PyPI](https://img.shields.io/pypi/v/inference-client)](https://pypi.org/project/inference-client/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/inference-client)](https://pypi.org/project/inference-client/)
[![PyPI - License](https://img.shields.io/pypi/l/inference-client)](https://pypi.org/project/inference-client/)

Inference-Client is a Python library that allows you to interact with the [Jina AI Inference](https://cloud.jina.ai/user/inference). 
It provides a simple and intuitive API to perform various tasks such as image captioning, encoding, ranking, visual 
question answering (VQA), and image upscaling.

The current version of Inference Client includes methods to call the following tasks:

📷 **Caption**: Generate captions for images 

📈 **Encode**: Encode data into embeddings using various models 

🔍 **Rank**: Re-rank cross-modal matches according to their joint likelihood

🆙 **Upscale**: Increasing the resolution while preserving the quality and details

🤔 **VQA**: Answer questions related to images 


## Installation

Inference Client is available on PyPI and can be installed using pip:

```bash
pip install inference-client
```

## Getting Started

Before using the Inference-Client, please create an inference on [Jina AI Cloud](https://cloud.jina.ai/user/inference).

After the inference is created and the status is "Serving", you can use the Inference-Client to connect to it.
This could take a few minutes, depending on the model you selected.

### Client Initialization

To use the Inference-Client, you first need to import the `Client` class and create a new instance of it. 

```python
from inference_client import Client

client = Client(token='<your auth token>')
```

You will need to provide your access token when creating the client. The token can be generated at the [Jina AI Cloud](https://cloud.jina.ai/settings/tokens), or via CLI as described in [this guide](https://docs.jina.ai/jina-ai-cloud/login/#create-a-new-pat):
```bash
jina auth token create <name of PAT> -e <expiration days>
```

You can then use the `get_model` method of the `Client` object to get a specific model.

```python
model = client.get_model('<model of your selection>')
```
You can connect to as many inference models as you want once they have been created on Jina AI Cloud, and you can use them for multiple tasks.

## Performing tasks

Now that you have connected to the models, you can use them to perform the tasks they support.

### Image Captioning

The `caption` method of the `Model` object takes an image as input and returns a caption as output.

```python
image = 'path/to/image.jpg'
caption = model.caption(image=image)
```

### Encoding

The `encode` method of the `Model` object takes text or image data as input and returns an embedding as output.

```python
text = 'a sentence describing the beautiful nature'
embedding = model.encode(text=text)

# OR
image = 'path/to/image.jpg'
embedding = model.encode(image=image)
```

### Ranking

The `rank` method of the `Model` object takes a text or image data as query and a list of candidates as input and returns a list of reordered candidates as well as their scores as output.

```python
candidates = [
    'an image about dogs',
    'an image about cats',
    'an image about birds',
]
image = 'path/to/image.jpg'
result = model.rank(image=image, candidates=candidates)
```

### Image Upscaling

The `upscale` method of the `Model` object takes an image and optional configurations as input, and returns the upscaled image bytes as output.

```python
image = 'path/to/image.jpg'
result = model.upscale(image=image, output_path='upscaled_image.png', scale='800:600')
```

### Visual Question Answering (VQA)

The `vqa` method of the `Model` object takes an image and a question as input and returns an answer as output.

```python
image = 'path/to/image.jpg'
question = 'Question: What is the name of this place? Answer:'
answer = model.vqa(image=image, question=question)
```

## Advanced Usage

In addition to the basic usage, the Inference-Client also supports advanced features such as handling DocumentArray inputs, customizing the task parameters, and more. 
Please refer to the [official documentation](https://jina.readme.io/docs/inference) for more details.

## Support

- Join our [Discord community](https://discord.jina.ai) and chat with other community members about ideas.
- Watch our [Engineering All Hands](https://youtube.com/playlist?list=PL3UBBWOUVhFYRUa_gpYYKBqEAkO4sxmne) to learn Jina's new features and stay up-to-date with the latest AI techniques.
- Subscribe to the latest video tutorials on our [YouTube channel](https://youtube.com/c/jina-ai)

## License

Inference-Client is backed by [Jina AI](https://jina.ai) and licensed under [Apache-2.0](./LICENSE). 