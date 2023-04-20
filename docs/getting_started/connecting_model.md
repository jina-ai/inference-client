## Connecting to Models

Once you have initialized the Client object, you can connect to the models you want to use by calling the `get_model` method, which takes the name of the model as it appears in Jina AI Cloud as an argument.

```python
from inference_client import Client

client = Client()
clip_model = client.get_model('ViT-B-32::openai')
blip_model = client.get_model('Salesforce/blip2-flan-t5-xl')
```

As example, the above code connects to the CLIP model named "ViT-B-32::openai" and the Blip model named "Salesforce/blip2-flan-t5-xl" on Jina AI Cloud.
