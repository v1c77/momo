# -*- coding: utf-8 -*-
import os

# base
CORE_STATIC_PATH = os.path.join(os.path.dirname(__file__), 'static')

# config
APP_CONFIG_PATH = 'app.yaml'
CORE_CONFIG_PATH = 'env.yaml'
GOGO_APP_NAME_PREFIX = 'gogo.'
ENV_VOS_PATH = 'gogo.vos.env'

PlACE_HODER = '-'

DEFAULT_WORKER_CONNECTIONS_GRPC = 1000
DEFAULT_WORKER_NUMBER_DEV = 1
DEFAULT_WORKER_NUMBER_PROD = 2
DEFAULT_WORKER_TIMEOUT = 30

REGISTRY_ENABLED = True

APP_TYPE_GRPC = 'thrift'
