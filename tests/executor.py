import io
import os

import numpy as np
from docarray import DocumentArray
from jina import Executor, requests
from PIL import Image


def calculate_output_size(input_w, input_h, model_w, model_h, target_w, target_h):
    output_w = input_w if target_w == 0 else target_w
    output_h = input_h if target_h == 0 else target_h

    if target_w < 0 and target_h >= 1:
        aspect_ratio = model_w / model_h
        output_w = int(abs(target_h) * aspect_ratio)
        if output_w % abs(target_w) != 0:
            output_w += abs(target_w) - (output_w % abs(target_w))

    elif target_h < 0 and target_w >= 1:
        aspect_ratio = model_w / model_h
        output_h = int(abs(target_w) / aspect_ratio)
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

    @requests(on=['/image-to-image', '/text-to-image'])
    def something_to_image(self, docs, **kwargs):
        for doc in docs:
            parameters = kwargs.get('parameters', {})
            output_type = parameters.get('output_type', 'pil')
            matches = DocumentArray.empty(
                int(parameters.get('num_images_per_prompt', 1))
            )
            for m in matches:
                m.uri = f'{os.path.dirname(os.path.abspath(__file__))}/test.jpeg'
                m.load_uri_to_blob()
                if output_type == 'latent':
                    m.convert_blob_to_image_tensor()
                m.uri = None
            doc.matches = matches

    @requests(on='/upscale')
    def upscale(self, docs, **kwargs):
        # This dummy upscale function has a hardcoded output size of 8x the input size
        for doc in docs:
            if doc.tensor is not None:
                doc.convert_image_tensor_to_blob()
            original_image = Image.open(io.BytesIO(doc.blob))
            input_w, input_h = original_image.size
            parameters = kwargs.get('parameters', {})
            scale = parameters.get('scale', None)
            if scale is None:
                output_w, output_h = input_w * 8, input_h * 8
            else:
                scales = scale.split(':')
                output_w, output_h = calculate_output_size(
                    input_w,
                    input_h,
                    input_w * 8,
                    input_h * 8,
                    int(scales[0]),
                    int(scales[1]),
                )
            upscale_image = original_image.resize((output_w, output_h))
            img_byte_arr = io.BytesIO()
            upscale_image.save(
                img_byte_arr, format=doc.tags.get('image_format', 'jpeg')
            )
            doc.blob = img_byte_arr.getvalue()

            if doc.tags.get('image_format', 'jpeg') == 'jpeg':
                if quality := parameters.get('quality', None):
                    img_byte_arr = io.BytesIO()
                    im = Image.open(io.BytesIO(doc.blob))
                    im.save(img_byte_arr, format='jpeg', quality=int(quality))
                    doc.blob = img_byte_arr.getvalue()

    @requests(on='/vqa')
    def vqa(self, docs, **kwargs):
        for doc in docs:
            doc.tags['response'] = 'Yes, it is a cat'


class ErrorExecutor(Executor):
    @requests
    def foo(self, docs, **kwargs):
        raise NotImplementedError


if __name__ == '__main__':
    from jina import Document, DocumentArray, Flow

    with Flow().add(uses=DummyExecutor) as f:
        input = DocumentArray(
            [
                Document(
                    uri=f'{os.path.dirname(os.path.abspath(__file__))}/test.jpeg'
                ).load_uri_to_blob()
            ]
        )
        f.post(on='/upscale', inputs=input, return_results=True)
