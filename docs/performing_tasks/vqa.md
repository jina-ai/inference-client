# VQA

VQA (Visual Question Answering) is a task in computer vision and natural language processing that involves answering questions about visual content, such as images or videos. 
The task requires understanding both the visual content and the natural language query, and providing an accurate answer in natural language. 
VQA has many practical applications, such as image search engines and virtual assistants.

The `vqa` method of the `BaseClient` object takes an image and a question as input and returns an answer as output.
The image can be a path to an image file, bytes, or an array that represents an image.
Once again, you can also wrap the input using `DocArray`.

## Plain Input

To generate an answer for a plain image and question, you can assign the image path, bytes, or array to the `image` parameter of the `vqa` method, and the question to the `question` parameter:

```python
from PIL import Image
import numpy as np

# Path to image
image = "path/to/image.jpg"

# Image bytes
image = open("path/to/image.jpg", "rb").read()

# Image array
image = np.array(Image.open("path/to/image.jpg"))

# Question
question = "Question: What is the name of this place? Answer:"

answer = model.vqa(image=image, question=question)
```

```bash
Singapore
```

```{note}
Due to the restrictions of the model, the question must start with `Question:` and end with `Answer:`.
```

## DocumentArray Input

You can also wrap the image and question using `DocumentArray` as shown in the following example.
Note that the question is stored as "prompt" in the `tags` attribute of the `Document` object.

```python
from jina import DocumentArray, Document

doc = DocumentArray(
    [
        Document(
            uri="path/to/image.jpg",
            tags={"prompt": "Question: What is the name of this place? Answer:"},
        )
    ]
)

# A list of Document objects
answer = model.vqa(docs=[doc])

# A DocumentArray object
answer = model.vqa(docs=DoucmentArray([doc]))

print(answer[0].tags["response"])
```

```bash
Singapore
```

The result will be a `DocumentArray` object with the answer stored in the `response` in the `tags` attribute of each `Document` object.
You can refer to the [DocArray documentation](https://docarray.org/legacy-docs/) to learn more about how to construct an [image `Document`](https://docarray.org/legacy-docs/datatypes/image/).