# Encode

Encoding in machine learning transforms raw data into a format that can be efficiently processed by a model.
The resulting encoded data, called embeddings, allows machine learning models to learn from input data and make accurate predictions or decisions.

The `encode` method of the `BaseClient` object takes raw data as input and returns embeddings as output.
It allows a wide range of data types as input, including plain text, image URIs, image bytes, arrays that represent images, and DocumentArrays.

## Plain Input

### Plain Text

To encode plain text data, you can assign the text to the `text` parameter of the `encode` method:

```python
embedding = model.encode(text='Hello, world!')
```

```bash
[0.123, 0.456, 0.789, ...]
```

### Plain Image

You can also encode plain image data by assigning the image path, bytes, or array to the `image` parameter of the `encode` method:

```python
from PIL import Image
import numpy as np

embedding = model.encode(image='path/to/image.jpg')
embedding = model.encode(image=open('path/to/image.jpg', 'rb'))
embedding = model.encode(image=np.array(Image.open('path/to/image.jpg')))
```

```bash
[0.123, 0.456, 0.789, ...]
```

### A List of Plain Inputs

In addition to a single input, you can also encode a list of inputs by assigning the list to the `text` or `image` parameter of the `encode` method:

```python
embeddings = model.encode(text=['Hello, world!', 'Hello, Jina!'])
embeddings = model.encode(image=['path/to/image1.jpg', 'path/to/image2.jpg'])
```

```bash
[[0.123, 0.456, 0.789, ...]
 [0.987, 0.654, 0.321, ...]]
```

The result will be a high-dimensional array of embeddings, where each row represents the embedding of the corresponding input.

## DocumentArray Input

The `encode` method also supports `DocumentArray` inputs.
[DocArray](https://github.com/docarray/docarray) is a library for **representing, sending and storing multi-model data**, which is perfect for **Machine Learning applications**.
You can pass a `DocumentArray` object or a list of `Document` objects to the `encode` method to encode the data:

```python
from jina import DocumentArray, Document

# A list of three Documents
docs = [
    Document(text='Hello, world!'),
    Document(text='Hello, Jina!'),
    Document(text='Hello, Goodbye!'),
]

# A DocumentArray containing three text Documents
docs = DocumentArray(
    [
        Document(text='Hello, world!'),
        Document(text='Hello, Jina!'),
        Document(text='Hello, Goodbye!'),
    ]
)

# A DocumentArray containing three image Documents
docs = DocumentArray(
    [
        Document(uri='path/to/image1.jpg'),
        Document(uri='path/to/image2.jpg').load_uri_to_blob(),
        Document(uri='path/to/image3.jpg').load_uri_to_image_tensor(),
    ]
)

result = model.encode(docs=docs)
print(result.embeddings)
```

```bash
[[0.123, 0.456, 0.789, ...]
 [0.987, 0.654, 0.321, ...]
 [0.111, 0.222, 0.333, ...]]
```

The result will be a `DocumentArray` object with the embedding of each input stored in the `embedding` attribute of each `Document` object.
You can refer to the [DocArray documentation](https://docarray.org/legacy-docs/) to learn more about how to construct a [text `Document`](https://docarray.org/legacy-docs/datatypes/text/) or an [image `Document`](https://docarray.org/legacy-docs/datatypes/image/).