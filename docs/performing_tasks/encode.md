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