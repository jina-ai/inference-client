# Getting Started

To get started with the Inference Client, you'll need to import the `Client` class from the `inference_client` package:

```python
from inference_client import Client
```

## Client Initialization

We provide various ways to initialize the Client object, depending on your use case. 

### Using a Jina AI Cloud authentication token

The simplest way is to initialize the Client object with your Jina AI Cloud authentication token:

```python
from inference_client import Client

client = Client(token='<your auth token>')
```
The token can be generated at the [Jina AI Cloud](https://cloud.jina.ai/settings/tokens), or via CLI as described in [this guide](https://docs.jina.ai/jina-ai-cloud/login/#create-a-new-pat):

```bash
jina auth token create <name of PAT> -e <expiration days>
```

```{warning}
Please note that the token will only be shown once, so make sure to save it somewhere safe.
```

### Using environment variables

You can also initialize the Client object by setting the `JINA_AUTH_TOKEN` environment variable.
This is convenient if you want to use the same token across multiple scripts and is the recommended way to initialize the Client object in production environments.

```python
import os
from inference_client import Client

os.environ['JINA_AUTH_TOKEN'] = '<your auth token>'
client = Client()
```

### Logging in via Web UI

For convenience, you can also use the Jina AI Cloud Web UI to log in and initialize the Client object. 
This will set the `JINA_AUTH_TOKEN` environment variable for you.
This is suitable for development environments, but not recommended for production environments since the session will expire after a certain amount of time.

```python
from inference_client import Client

client = Client()
```

## Connecting to Models

Once you have initialized the Client object, you can connect to the models you want to use by calling the `get_model` method, which takes the name of the model as it appears in Jina AI Cloud as an argument.

```python