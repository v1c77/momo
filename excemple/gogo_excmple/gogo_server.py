# -*- coding: utf-8 -*-

from gogo.service import Service
from . import greater_sdk

service = Service(
    timeout=3 * 1000,
    service_name='GoGo_Service',

    user_exc=greater_sdk.GeUserException,
    system_exc=greater_sdk.GeSystemException,
    unknown_exc=greater_sdk.GeUnknownException,
    error_code=greater_sdk.GeErrorCode,
)


class Dispatcher(object):
    pass


service.register(Dispatcher)

