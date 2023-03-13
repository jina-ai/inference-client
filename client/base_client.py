class BaseClient(object):
    def __init__(self, host, port, model_name, model_version, timeout=10):
        self.host = host
        self.port = port
        self.model_name = model_name

    def encode(self, data):
        print('BaseClient.encode()')
        pass

    async def aencode(self, data):
        print('BaseClient.aencode()')
        pass

    def caption(self, data):
        print('BaseClient.caption()')
        pass

    async def acaption(self, data):
        print('BaseClient.acaption()')
