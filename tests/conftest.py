import pytest


@pytest.fixture(scope='session', params=['clip', 'blip2'])
def make_model_name_and_host(request):
    if request.param == 'blip2':
        return (
            'Salesforce/blip2-opt-2.7b',
            'grpcs://crucial-gazelle-779d1c8739-grpc.wolf.jina.ai',
        )
    else:
        return 'ViT-B-32::openai', 'grpcs://api.clip.jina.ai:2096'
