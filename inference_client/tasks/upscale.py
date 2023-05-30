import os
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
        scale: Optional[str] = None,
        image_format: Optional[str] = None,
        output_path: Optional[str] = None,
        **kwargs,
    ):
        """
        Upscale plain input images. Return an image bytes in the format of `image_format`. If `output_path` is
        provided, the image will also be saved to the specified path.

        :param image: the image to upscale, can be a `ndarray`, 'bytes' or uri of the image
        :param scale: the scale of the output image, if not provided, the image of the model output will be used. The
                scale should be in the format of `width:height`, e.g. `100:200`. Both width and height should be
                integers. If the width is 0, the input width is used for the output. If the height is 0, the input
                height is used for the output. If one and only one of the values is -n with n >= 1, the scale filter
                will use a value that maintains the aspect ratio of the input image, calculated from the other specified
                dimension. After that it will, however, make sure that the calculated dimension is divisible by n and
                adjust the value if necessary. If both values are -n with n >= 1, the behavior will be identical to both
                values being set to 0 as previously detailed. Default: None.
        :param image_format: the format of the output image, could be either `jpeg` or `png`. If not provided, the
                same format as the input image will be used. Default: None.
        :param output_path: the complete file path to save the output image if provided in addition to returning it.
                The path should end with the file extension of the output format, and could be either '.jpeg' or '.png'.
                If both `output_path` and `image_format` are provided, the `image_format` will be ignored, and the
                output image will be saved in the format of the file extension of `output_path`. Default: None.
        :param kwargs: additional arguments to pass to the model.
        """
        ...

    @overload
    def upscale(
        self,
        *,
        docs: Union[Iterable['Document'], 'DocumentArray'],
        scale: Optional[str] = None,
        **kwargs,
    ):
        """
        Upscale image documents. The result will be stored in the `blob` attribute of the document in the format of
        `image_format` specified in the `tags` attribute. If `output_path` is provided in the `tags` attribute, the
        image will also be saved to the specified path.

        :param docs: the image documents to upscale. The `image_format` and `output_path` should be provided in the
                `tags` attribute of each document. The `image_format` specifies the format of the output image, could
                be either `jpeg` or `png`. The `output_path` specifies the complete file path to save the output image
                if provided in addition to returning it. The path should end with the file extension of the output
                format, and could be either '.jpeg' or '.png'. If both `output_path` and `image_format` are provided,
                the `image_format` will be ignored, and the output image will be saved in the format of the file
                extension of `output_path`.
        :param scale: the scale of the output image, if not provided, the image of the model output will be used. The
                scale should be in the format of `width:height`, e.g. `100:200`. Both width and height should be
                integers. If the width is 0, the input width is used for the output. If the height is 0, the input
                height is used for the output. If one and only one of the values is -n with n >= 1, the scale filter
                will use a value that maintains the aspect ratio of the input image, calculated from the other specified
                dimension. After that it will, however, make sure that the calculated dimension is divisible by n and
                adjust the value if necessary. If both values are -n with n >= 1, the behavior will be identical to both
                values being set to 0 as previously detailed. Default: None.
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
        image_format: Optional[str] = None,
        output_path: Optional[str] = None,
        **kwargs,
    ):
        """
        Upscale an image or a set of image documents using a pre-trained model.

        :param docs: the image documents to upscale. The `image_format` and `output_path` should be provided in the
                `tags` attribute of each document. The `image_format` specifies the format of the output image, could
                be either `jpeg` or `png`. The `output_path` specifies the complete file path to save the output image
                if provided in addition to returning it. The path should end with the file extension of the output
                format, and could be either '.jpeg' or '.png'. If both `output_path` and `image_format` are provided,
                the `image_format` will be ignored, and the output image will be saved in the format of the file
                extension of `output_path`. Default: None.
        :param image: the image to upscale, can be a `ndarray`, 'bytes' or uri of the image. Default: None.
        :param scale: the scale of the output image, if not provided, the image of the model output will be used. The
                scale should be in the format of `width:height`, e.g. `100:200`. Both width and height should be
                integers. If the width is 0, the input width is used for the output. If the height is 0, the input
                height is used for the output. If one and only one of the values is -n with n >= 1, the scale filter
                will use a value that maintains the aspect ratio of the input image, calculated from the other specified
                dimension. After that it will, however, make sure that the calculated dimension is divisible by n and
                adjust the value if necessary. If both values are -n with n >= 1, the behavior will be identical to both
                values being set to 0 as previously detailed. Default: None.
        :param image_format: the format of the output image, could be either `jpeg` or `png`. If not provided, the
                same format as the input image will be used. Default: None.
        :param output_path: the complete file path to save the output image if provided in addition to returning it.
                The path should end with the file extension of the output format, and could be either '.jpeg' or '.png'.
                If both `output_path` and `image_format` are provided, the `image_format` will be ignored, and the
                output image will be saved in the format of the file extension of `output_path`. Default: None.
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

        if kwargs.get('docs', None) is not None:
            if kwargs.get('image', None) is not None:
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

        elif kwargs.get('image', None) is not None:
            content_type = 'plain'
            image_content = kwargs.pop('image')
            if isinstance(image_content, (str, bytes, numpy.ndarray)):
                image_doc = load_plain_into_document(image_content, mime_type='image')

                if kwargs.get('output_path', None) is not None:
                    output_path = kwargs.pop('output_path')
                    image_format = os.path.splitext(output_path)[1].lower()
                    if image_format not in ('.jpeg', '.jpg', '.png'):
                        raise ValueError(
                            'Output path should end with either `.jpeg` or `.png`.'
                        )
                    image_doc.tags['output_path'] = output_path

                if kwargs.get('image_format', None) is not None:
                    image_format = kwargs.pop('image_format').lower()
                    if image_format not in ('jpeg', 'jpg', 'png'):
                        raise ValueError('Output format should be either jpeg or png.')
                    image_doc.tags['image_format'] = image_format

                payload.update(inputs=DocumentArray([image_doc]))
                payload.update(total_docs=1)
            else:
                raise ValueError('Only single image input is supported.')

        else:
            raise ValueError('Please provide either image or docs input.')

        if kwargs.get('scale', None) is not None:
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
        for doc in result:
            if output_path := doc.tags.get('output_path'):
                with open(output_path, 'wb') as f:
                    f.write(doc.blob)

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
