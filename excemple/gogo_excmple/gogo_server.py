# -*- coding: utf-8 -*-
import time

from concurrent import futures

from gogo.service import Service
from . import hello_bro_pb2

service = Service(
    timeout=3 * 1000,
)


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


# TODO(vici)
class Dispatcher(server):
    def SayHello(self, request, context):
        pool = futures.ThreadPoolExecutor(max_workers=2)
        pool.submit(func_cost)

        return hello_bro_pb2.HelloReply(
            message='Hello, %s!' % request.name, by='gogo')


service.register(Dispatcher)
