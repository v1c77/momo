# -*- coding: utf-8 -*-

from __future__ import print_function

import grpc

from test_grpc.greater_sdk import greater_pb2
from test_grpc.greater_sdk import greater_pb2_grpc


def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    channel = grpc.insecure_channel('localhost:1994')
    stub = greater_pb2_grpc.GreeterStub(channel)
    response = stub.SayHello(greater_pb2.HelloRequest(name='you'))
    print("Greeter client received: " + response.message)


if __name__ == '__main__':
    run()
