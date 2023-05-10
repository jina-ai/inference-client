---
title: Client Initialization Title Test
category: 645b1e71f6dfd506df5acc01
hidden: false
---

# Client Initialization

To get started with the Inference Client, you'll need to import the `Client` class from the `inference_client` package:

```python
from inference_client import Client
```

We provide various ways to initialize the `Client` object, depending on your use case. 

## Using a Jina AI Cloud authentication token

The simplest way to initialize the `Client` object is to pass your Jina AI Cloud authentication token as a parameter when creating an instance of the `Client` class:

```python
client = Client(token='<your auth token>')
```

You can generate an authentication token in the [Jina AI Cloud UI](https://cloud.jina.ai/settings/tokens) by going to the Tokens section in your account settings. 
Alternatively, you can generate an authentication token using the [Jina CLI](https://docs.jina.ai/jina-ai-cloud/login/#create-a-new-pat):

```bash
jina auth token create <name of PAT> -e <expiration days>
```

```{warning}
Please note that the token will only be shown once, so make sure to save it somewhere safe.
```

## Using environment variables

Another way to initialize the `Client` object is to set the `JINA_AUTH_TOKEN` environment variable to your authentication token before creating an instance of the `Client` class:

```python
import os
from inference_client import Client

os.environ['JINA_AUTH_TOKEN'] = '<your auth token>'
client = Client()
```

Setting the `JINA_AUTH_TOKEN` environment variable is a convenient way to use the same authentication token across multiple scripts, and is the recommended way to initialize the `Client` object in production environments.

## Logging in via Web UI

For convenience, you can also use the Jina AI Cloud Web UI to log in and initialize the `Client` object. 
This will set the `JINA_AUTH_TOKEN` environment variable for you.
This is suitable for development environments, but not recommended for production environments since the session will expire after a certain amount of time.

```python
client = Client()
```