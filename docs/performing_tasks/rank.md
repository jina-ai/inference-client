# Rank

In machine learning, ranking is the process of ordering a set of items based on their relevance to a specific query. 
It is often used in information retrieval, recommendation systems, and search engines to provide a personalized and optimized user experience. 
The goal of ranking is to present the most relevant items to the user, based on their preferences and behavior.

The `rank` method of the `BaseClient` object takes a reference as query and a list of candidates as input and returns a list of reordered candidates as well as their scores as output.
You can also wrap the reference and candidates using `DocArray`.

## Plain Input

When using plain input with the rank method, the reference and candidates can be passed as text or image data. 
The reference should be passed as a single data entry, while the candidates can be passed as a list of data entries. 
Similar to the `encode` method, each data entry can be a string, a path to an image, bytes of an image, or an array that represents an image.

```python
reference = 'path/to/image.jpg'
candidates = [
    'an image about dogs',
    'an image about cats',
    'an image about birds',
]

result = model.rank(reference=reference, candidates=candidates)
```

```bash
[('an image about cats', {'clip_score_cosine': {'value': 0.3,'op_name': 'cosine',},'clip_score': {'value': 0.6, 'op_name': 'softmax'},},
('an image about dogs', {'clip_score_cosine': {'value': 0.2,'op_name': 'cosine',},'clip_score': {'value': 0.4, 'op_name': 'softmax'},},
('an image about birds', {'clip_score_cosine': {'value': 0.1,'op_name': 'cosine',},'clip_score': {'value': 0.2, 'op_name': 'softmax'},}]
```

The result will be a list of tuples, where each tuple contains the candidate and its score.

## DocumentArray Input

When using `DocumentArray` input with the rank method, the reference and candidates can be passed as `DocumentArray` objects or lists of `Document` objects.

First construct a cross-modal `Document` where the root contains an image and `.matches` contain sentences to rerank. 
You can also construct text-to-image rerank as below:

```python
from jina import DocumentArray, Document

doc = Document(
    text='a photo of conference room',
    matches=[
        Document(uri='path/to/image1.jpg'),
        Document(uri='path/to/image2.jpg'),
        Document(uri='path/to/image3.jpg'),
    ],
)
```

Then pass the `Document` to the rank method:

```python
# A list of Document objects
result = model.rank(docs=[doc])

# A DocumentArray object
result = model.rank(docs=DocumentArray([doc]))
```

The result will be a DocumentArray object with the same structure as the input, but with the `.matches` reordered and scored.
You can access the matches and their scores using the `matches` and `scores` attributes of the `Document` object like below:

```python
for match in result[0].matches:
    print(match.uri, match.scores)
```

```bash
path/to/image2.jpg {'clip_score_cosine': {'value': 0.3,'op_name': 'cosine',},'clip_score': {'value': 0.6, 'op_name': 'softmax'},}
path/to/image1.jpg {'clip_score_cosine': {'value': 0.2,'op_name': 'cosine',},'clip_score': {'value': 0.4, 'op_name': 'softmax'},}
path/to/image3.jpg {'clip_score_cosine': {'value': 0.1,'op_name': 'cosine',},'clip_score': {'value': 0.2, 'op_name': 'softmax'},}
```

You can refer to the [DocArray documentation](https://docarray.org/legacy-docs/) to learn more about how to construct a [text `Document`](https://docarray.org/legacy-docs/datatypes/text/) or an [image `Document`](https://docarray.org/legacy-docs/datatypes/image/).
You can also learn more about how to construct a [Nested Structure `Document`](https://docarray.org/legacy-docs/fundamentals/document/nested/)
