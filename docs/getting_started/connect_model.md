## Connecting to Models

Once you have initialized the Client object, you can connect to the models you created at [Jina AI Cloud](https://cloud.jina.ai) by calling the `get_model` method, which takes the name of the model as it appears in Jina AI Cloud as an argument.

```python
from inference_client import Client

client = Client()
clip_model = client.get_model('ViT-B-32::openai')
blip_model = client.get_model('Salesforce/blip2-flan-t5-xl')
```

The `get_model` method returns a `Model` object that provides a high-level interface for interacting with the model. 
The `Model` object includes methods such as encoding data using the model, as well as other model-specific functionality. 
As example, the above code connects to the CLIP model named "ViT-B-32::openai" and the Blip model named "Salesforce/blip2-flan-t5-xl" on Jina AI Cloud.

Once you have a `Model` object, you can use its methods to interact with the model. 
For example, the `clip_model` object has an `encode` method that takes an image path as an argument and returns an embedding vector:

```python
# Encode an image using the CLIP model
embedding = clip_model.encode(image='path/to/image.jpg')
```

Similarly, the `blip_model` object has a `caption` method that takes an image path as an argument and generates a caption that describes the image:

```python
caption = blip_model.caption(image='path/to/image.jpg')
```

You can connect to as many inference models as you want once they have been created on Jina AI Cloud, and you can integrate them into your application for multiple and complex tasks.