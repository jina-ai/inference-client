======================
:fab:`python` Python API
======================

This section includes the API documentation from the `inference_client` codebase, as extracted from the `docstrings <https://peps.python.org/pep-0257/>`_ in the code.

:mod:`inference_client.client.Client` - Client
--------------------

.. currentmodule:: inference_client.client.Client

.. autosummary::
   :nosignatures:
   :template: class.rst

   inference_client.client.Client.__init__
   inference_client.client.Client.get_models


:mod:`inference_client.base.BaseClient` - BaseClient
--------------------

.. currentmodule:: inference_client.base.BaseClient

.. autosummary::
   :nosignatures:
   :template: class.rst

   inference_client.base.BaseClient.encode
   inference_client.base.BaseClient.rank
   inference_client.base.BaseClient.caption
   inference_client.base.BaseClient.vqa
   inference_client.base.BaseClient._iter_doc
   inference_client.base.BaseClient._get_post_payload
   inference_client.base.BaseClient._post
   inference_client.base.BaseClient._unboxed_result