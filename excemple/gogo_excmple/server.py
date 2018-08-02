# -*- coding: utf-8 -*-

from concurrent import futures
import time

import grpc

from .greater_sdk import greater_pb2
from .greater_sdk import greater_pb2_grpc


_ONE_DAY_IN_SECONDS = 60 * 60 * 24


def func_cost():
    print('c is runing.!!!')
    start = time.time()
    end = start + 10
    i = 1

    while i > 0:
        i += 1
        if time.time() > end:
            break
    print('down')


class Greeter(greater_pb2_grpc.GreeterServicer):

    def SayHello(self, request, context):
        pool = futures.ThreadPoolExecutor(max_workers=2)
        pool.submit(func_cost)

        return greater_pb2.HelloReply(message='Hello, %s!' % request.name,
                                      name=request.name)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=65))
    greater_pb2_grpc.add_GreeterServicer_to_server(Greeter(), server)
    server.add_insecure_port('[::]:1994')
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    print("start at 1994")
    serve()
