import pytest
from jina import Flow, helper

from inference_client.model import Model

from .executor import DummyExecutor, ErrorExecutor


@pytest.fixture(scope='session')
def port_generator():
    generated_ports = set()

    def random_port():
        port = helper.random_port()
        while port in generated_ports:
            port = helper.random_port()
        generated_ports.add(port)
        return port

    return random_port


@pytest.fixture(scope='session')
def make_flow(port_generator):
    f = Flow(port=port_generator()).add(name='dummy', uses=DummyExecutor)
    with f:
        yield f


@pytest.fixture(scope='session')
def make_client(make_flow):
    return Model(
        model_name='dummy-model',
        token='valid_token',
        host=f'grpc://0.0.0.0:{make_flow.port}',
    )


@pytest.fixture(scope='session')
def make_error_flow(port_generator):
    f = Flow(port=port_generator()).add(name='error', uses=ErrorExecutor)
    with f:
        yield f


@pytest.fixture(scope='session')
def make_error_client(make_error_flow):
    return Model(
        model_name='error-model',
        token='valid_token',
        host=f'grpc://0.0.0.0:{make_error_flow.port}',
    )
