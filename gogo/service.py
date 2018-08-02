# -*- coding: utf-8 -*-
import logging
import time

import functools

from gogo.ctx import g
# TODO(vici) db session manager for mongo
# TODO(vici) config module
from gogo.vos.config import load_app_config
# TODO(vici) request data meta
# TODO(vici) env module
from gogo.consts import PlACE_HODER
from gogo.log import make_meta_dict
from utils import random_string, filter_apis_to_shield


# TODO(vici) hook event import for diff handlers of gunicorn


def _save_mate_data(func):
    """the id and other info for each request"""
    @functools.wraps(func)
    def wrapper(*args):
        # default values
        seq = PlACE_HODER
        rid = g.get_call_meta('tracker_request_id')
        seq = g.get_call_meta('tracker_seq')
        meta = g.get_call_meta('meta', {})
        if rid == PlACE_HODER:
            rid = 'fake_' + random_string(8)

        g.clear_api_ctx()  # clean exist local info

        g.set_call_meta('call_time', time.time())
        g.set_call_meta('method', func.func_name)
        g.set_call_meta('args', args)
        g.set_call_meta('request_id', rid)
        g.set_call_meta('seq', seq)
        g.set_call_meta('meta', meta)
        make_meta_dict("meta", meta)

        return func(*args)
    return wrapper

# TODO(vici) add profile handler
# TODO(vici) with db session tracker


class Service(dict):
    PROHIBITED_ATTRS = {}  # attrs can not visit
    REQUESTED_PROCESSORS = {}  # processors for gogo server

    __getattr__ = dict.__getitem__

    def __setattr__(self, name, value):
        return super(Service, self).__setattr__(name, value)

    def __init__(self, *args, **kwargs):
        super(Service, self).__init__(*args, **kwargs)
        self['__apis_to_shield'] = {}  # ???
        self.logger = logging.getLogger(__name__)

        # TODO(vici) grpc dispatcher cls
        self.grpc_service_dispatcher_cls = None
        self.grpc = self.get('grpc', None) or \
            load_app_config().get_grpc_module()

        if self.get('grpc', None) and self.get('service_name', None):
            self['__apis_to_shield'].update(filter_apis_to_shield(
                self.grpc, self.service_name))  # ???

        for exc_name in ('user_exc', 'system_exc', 'unknown_exc'):
            if exc_name not in self:
                raise RuntimeError(
                    "Service should contain following exception definitions: "
                    "'user_exc', 'system_exc', 'unknown_exc'"
                )

        if 'error' not in self:
            self.error = self.user_exc, self.system_exc, self.unknown_exc

        # if service not has `name` attr, load from app.yaml
        if 'name' not in self:
            self.name = load_app_config().app_name

        else:
            if self.name != load_app_config().app_name:
                msg = ("`app name` in `app.yaml` is different from that in "
                       "`service.py`, and we take `app name` in `app.yaml` "
                       "as standard.")
                self.logger.error(msg)
                raise RuntimeError(msg)

        self.timeout = self.get("timeout", DEFAULT_SOFT_TIMEOUT)
        self.api_hard_timeout = self.get("api_hard_timeout",
                                         DEFAULT_HARD_TIMEOUT)
        # TODO(vici) if add soa
        # self._config_manager = get_soa_config()
        self.__processors = []
        for proc_cls in self.REQUESTED_PROCESSORS:
            self.__processors.append(proc_cls(self))
