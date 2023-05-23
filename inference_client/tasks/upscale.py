from typing import TYPE_CHECKING, Iterable, Optional, Union, overload

import numpy
from docarray import Document, DocumentArray
from jina import Client

from .helper import get_base_payload, iter_doc, load_plain_into_document

if TYPE_CHECKING:
    from docarray.typing import ArrayType


class UpscaleMixin:
    """
    Mixin class for up-scaling image.
    """

    token: str
    client: Client

    @overload
    def upscale(
        self,
        *,
        image: Union[str, bytes, 'ArrayType'],
        scale: Optional[str],
        **kwargs,
    ):
        """
        Upscale plain input images. The result will be an image bytes.

        :param image: the image to upscale, can be a `ndarray`, 'bytes' or uri of the image
        :param scale: the scale of the output image, if not provided, the image of the model output will be used. The
                scale should be in the format of `width:height`, e.g. `100:200`. Both width and height should be
                integers. If the width is 0, the input width is used for the output. If the height is 0, the input
                height is used for the output. If one and only one of the values is -n with n >= 1, the scale filter
                will use a value that maintains the aspect ratio of the input image, calculated from the other specified
                dimension. After that it will, however, make sure that the calculated dimension is divisible by n and
                adjust the value if necessary. If both values are -n with n >= 1, the behavior will be identical to both
                values being set to 0 as previously detailed.
        :param kwargs: additional arguments to pass to the model
        """
        ...

    @overload
    def upscale(
        self,
        *,
        docs: Union[Iterable['Document'], 'DocumentArray'],
        scale: Optional[str],
        **kwargs,
    ):
        """
        Upscale image documents. The result will be stored in the `blob` attribute of the document.

        :param docs: the image documents to upscale
        :param scale: the scale of the output image, if not provided, the image of the model output will be used. The
                scale should be in the format of `width:height`, e.g. `100:200`. Both width and height should be
                integers. If the width is 0, the input width is used for the output. If the height is 0, the input
                height is used for the output. If one and only one of the values is -n with n >= 1, the scale filter
                will use a value that maintains the aspect ratio of the input image, calculated from the other specified
                dimension. After that it will, however, make sure that the calculated dimension is divisible by n and
                adjust the value if necessary. If both values are -n with n >= 1, the behavior will be identical to both
                values being set to 0 as previously detailed.
        :param kwargs: additional arguments to pass to the model
        """
        ...

    @overload
    def upscale(
        self,
        *,
        docs: Optional[Union[Iterable['Document'], 'DocumentArray']] = None,
        image: Optional[Union[str, bytes, 'ArrayType']] = None,
        scale: Optional[str] = None,
        **kwargs,
    ):
        """
        Upscale an image or a set of image documents using a pre-trained model.

        :param docs: the image documents to upscale. Defaults to None.
        :param image: the image to upscale, can be a `ndarray`, 'bytes' or uri of the image. Defaults to None.
        :param scale: the scale of the output image, if not provided, the image of the model output will be used. The
                scale should be in the format of `width:height`, e.g. `100:200`. Both width and height should be
                integers. If the width is 0, the input width is used for the output. If the height is 0, the input
                height is used for the output. If one and only one of the values is -n with n >= 1, the scale filter
                will use a value that maintains the aspect ratio of the input image, calculated from the other specified
                dimension. After that it will, however, make sure that the calculated dimension is divisible by n and
                adjust the value if necessary. If both values are -n with n >= 1, the behavior will be identical to both
                values being set to 0 as previously detailed.
        :param kwargs: additional arguments to pass to the model.
        """
        ...

    def upscale(self, **kwargs):
        """
        Upscale the image documents using the model.

        :param kwargs: additional arguments to pass to the model.
        :return: upscaled image.
        """
        payload, content_type = self._get_upscale_payload(**kwargs)
        result = self.client.post(**payload)
        return self._unbox_upscale_result(
            result=result,
            content_type=content_type,
        )

    def _get_upscale_payload(self, **kwargs):
        payload = get_base_payload('/upscale', self.token, **kwargs)

        if 'docs' in kwargs:
            if 'image' in kwargs:
                raise ValueError(
                    'More than one input type provided. Please provide only docs or image input.'
                )
            content_type = 'docarray'
            total_docs = (
                len(kwargs.get('docs'))
                if hasattr(kwargs.get('docs'), '__len__')
                else None
            )
            payload.update(total_docs=total_docs)
            payload.update(inputs=iter_doc(kwargs.pop('docs')))

        elif 'image' in kwargs:
            content_type = 'plain'
            image_content = kwargs.pop('image')
            if isinstance(image_content, (str, bytes, numpy.ndarray)):
                image_doc = load_plain_into_document(image_content, mime_type='image')
                payload.update(inputs=DocumentArray([image_doc]))
                payload.update(total_docs=1)
            else:
                raise ValueError('Only single image input is supported.')

        else:
            raise ValueError('Please provide either image or docs input.')

        if 'scale' in kwargs:
            scale = kwargs.pop('scale')
            self._scale_checker(scale)
            if parameters := payload.get('parameters'):
                parameters.update(scale=scale)
            else:
                payload.update(parameters={'scale': scale})

        return payload, content_type

    def _unbox_upscale_result(
        self,
        result: 'DocumentArray' = None,
        content_type: str = 'docarray',
    ):
        if content_type == 'plain':
            return result[0].blob
        else:
            return result

    def _scale_checker(self, scale: str = None):
        if not isinstance(scale, str):
            raise ValueError('Scale should be a string.')
        scales = scale.split(':')
        if len(scales) != 2:
            raise ValueError('Scale should be in the format of `width:height`.')
        try:
            int(scales[0]) and int(scales[1])
        except ValueError:
            raise ValueError('Both width and height should be integers.')
