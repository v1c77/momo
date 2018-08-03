# -*- coding: utf-8 -*-

from __future__ import print_function

import grpc

from excemple.gogo_excmple import hello_bro_pb2
from excemple.gogo_excmple import hello_bro_pb2_grpc


def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    channel = grpc.insecure_channel('localhost:1994')
    stub = hello_bro_pb2_grpc.BroStub(channel)
    response = stub.SayHello(hello_bro_pb2.HelloRequest(name='you'))
    print("Greeter client received: " + response.message)


if __name__ == '__main__':
    run()
