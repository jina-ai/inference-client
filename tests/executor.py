import numpy as np
from jina import Executor, requests


class DummyExecutor(Executor):
    @requests(on='/caption')
    def caption(self, docs, **kwargs):
        for doc in docs:
            doc.tags['response'] = 'A image of something very nice'

    @requests(on='/encode')
    def encode(self, docs, **kwargs):
        for doc in docs:
            doc.embedding = np.random.random((512,))

    @requests(on='/rank')
    def rank(self, docs, **kwargs):
        for doc in docs:
            for m in doc.matches:
                m.scores['cosine'].value = np.random.random()
                m.scores['cosine'].op_name = 'cosine'
            final = sorted(
                doc.matches, key=lambda _m: _m.scores['cosine'].value, reverse=True
            )
            doc.matches = final

    @requests(on='/vqa')
    def vqa(self, docs, **kwargs):
        for doc in docs:
            doc.tags['response'] = 'Yes, it is a cat'
