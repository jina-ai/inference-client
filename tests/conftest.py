import pytest
from jina import Flow, helper

from .executor import DummyExecutor


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
