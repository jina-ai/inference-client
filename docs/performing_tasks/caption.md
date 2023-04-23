# Caption

Captioning is a computer vision task that involves generating a textual description of an image. 
It is a popular area of research in AI and has many practical applications, such as creating alt-text for images to improve accessibility, generating image descriptions for the visually impaired, and automatically tagging images for search and categorization purposes. 

The `caption` method of the `BaseClient` object takes an image as input and returns a caption as output.
The image can be a path to an image file, bytes, or an array that represents an image.
You can also wrap the image using `DocArray`.

## Plain Input

To generate a caption for a plain image, you can assign the image path, bytes, or array to the `image` parameter of the `caption` method:

```python
from PIL import Image
import numpy as np

caption = model.caption(image='path/to/image.jpg')
caption = model.caption(image=open('path/to/image.jpg', 'rb'))
caption = model.caption(image=np.array(Image.open('path/to/image.jpg')))
```

For example, the following image will generate a caption similar to 'the merlion statue in singapore at night':

<p align="center">
<img src="../_static/singapore.jpg" width="50%" />
</p>

## DocumentArray Input

The `caption` method also supports `DocumentArray` inputs.
[DocArray](https://github.com/docarray/docarray) is a library for **representing, sending and storing multi-model data**, which is perfect for **Machine Learning applications**.
You can pass a `DocumentArray` object or a list of `Document` objects to the `encode` method to encode the data:

```python
from jina import DocumentArray, Document

# A list of Document objects
docs = [Document(uri='path/to/image.jpg')]

# A DocumentArray object
docs = DocumentArray([Document(uri='path/to/image.jpg')])

result = model.caption(docs=docs)
print(result[0].tags['response'])
```

The result will be a `DocumentArray` object with the caption stored in the `response` in the `tags` attribute of each `Document` object.
You can refer to the [DocArray documentation](https://docarray.org/legacy-docs/) to learn more about how to construct an [image `Document`](https://docarray.org/legacy-docs/datatypes/image/).