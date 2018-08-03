# -*- coding: utf-8 -*-

from concurrent import futures
import traceback
import inspect
import time
import logging
import sys
import grpc

from excemple.gogo_excmple import hello_bro_pb2
from excemple.gogo_excmple import hello_bro_pb2_grpc


root = logging.getLogger()
root.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)

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


class Bro(hello_bro_pb2_grpc.BroServicer):

    def SayHello(self, request, context):
        # pool = futures.ThreadPoolExecutor(max_workers=2)
        # pool.submit(func_cost)
        root.info('trace')
        traceback.print_stack()
        # print(inspect.stack())

        return hello_bro_pb2.HelloReply(
            message='Hello, %s!' % request.name,
            by=request.name)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=65))
    hello_bro_pb2_grpc.add_BroServicer_to_server(Bro(), server)
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
