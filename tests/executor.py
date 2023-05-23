import numpy as np
from jina import Executor, requests


def calculate_output_size(input_w, input_h, target_w, target_h):
    output_w = input_w if target_w == 0 else target_w
    output_h = input_h if target_h == 0 else target_h

    if target_w < 0 and target_h >= 1:
        aspect_ratio = input_w / input_h
        output_w = int(aspect_ratio * abs(target_h))
        if output_w % abs(target_w) != 0:
            output_w += abs(target_w) - (output_w % abs(target_w))

    elif target_h < 0 and target_w >= 1:
        aspect_ratio = input_h / input_w
        output_h = int(aspect_ratio * abs(target_w))
        if output_h % abs(target_h) != 0:
            output_h += abs(target_h) - (output_h % abs(target_h))

    elif target_w < 0 and target_h < 0:
        output_w = input_w
        output_h = input_h

    return output_w, output_h


class DummyExecutor(Executor):
    @requests(on='/caption')
    def caption(self, docs, **kwargs):
        for doc in docs:
            doc.tags['response'] = 'A image of something very nice'

    @requests(on='/encode')
    def encode(self, docs, **kwargs):
        docs.embeddings = np.random.random((len(docs), 512))

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

    @requests(on='/upscale')
    def upscale(self, docs, **kwargs):
        # This dummy upscale function has a hardcoded output size of 8x the input size
        for doc in docs:
            if doc.blob:
                doc.convert_blob_to_image_tensor()
            input_w, input_h = doc.tensor.shape[1], doc.tensor.shape[0]
            parameters = kwargs.get('parameters', {})
            scale = parameters.get('scale', None)
            if scale is None:
                output_w, output_h = input_w * 8, input_h * 8
            else:
                scales = scale.split(':')
                output_w, output_h = calculate_output_size(
                    input_w, input_h, int(scales[0]), int(scales[1])
                )
            doc.tensor = np.random.random((output_h, output_w, 3))
            doc.convert_image_tensor_to_blob()

    @requests(on='/vqa')
    def vqa(self, docs, **kwargs):
        for doc in docs:
            doc.tags['response'] = 'Yes, it is a cat'


if __name__ == '__main__':
    from jina import Document, DocumentArray, Flow

    with Flow().add(uses=DummyExecutor) as f:
        input = DocumentArray(
            [Document(uri='https://picsum.photos/id/233/100').load_uri_to_blob()]
        )
        f.post(on='/upscale', inputs=input, return_results=True)
