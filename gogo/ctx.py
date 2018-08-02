# -*- coding: utf-8 -*-

import threading
from gogo.consts import PlACE_HODER
from gogo.vos import env


class ConnCtx(threading.local):
    def __init__(self):
        env.require('gevent_patched')
        super(ConnCtx, self).__init__()
        self.conn_ctx = {}
        self.call_meta_data = {}
        self.logging_meta = {}
        self.user_request_ctx = {}  # user data
        self.cached_return_ctx = {}

    def clear_api_ctx(self):
        self.call_meta_data.clear()
        self.logging_meta.clear()

    def clear_conn_ctx(self):
        self.conn_ctx.clear()

    def clear_user_request_ctx(self):
        self.user_request_ctx.clear()

    def clear_cached_return_ctx(self):
        self.cached_return_ctx.clear()

    def set_call_meta(self, key, value):
        self.call_meta_data[key] = value

    def set_conn_meta(self, key, value):
        self.conn_ctx[key] = value

    def set_user_request_ctx(self, key, value):
        self.user_request_ctx[key] = value

    def set_cached_return_ctx(self, key, value):
        self.cached_return_ctx[key] = value

    def get_call_meta(self, key, default=PlACE_HODER):
        return self.call_meta_data.get(key, default)

    def get_conn_meta(self, key, default=PlACE_HODER):
        return self.conn_ctx.get(key, default)

    def get_user_request_ctx(self, key):
        return self.user_request_ctx[key]

    def get_cached_return_ctx(self, key):
        return self.cached_return_ctx[key]

    def clear_all(self):
        self.clear_api_ctx()
        self.clear_conn_ctx()
        self.clear_user_request_ctx()
        self.clear_cached_return_ctx()

    def get_caller_id(self):
        # TODO(vici) if updated to soa use this func to tarce back soa
        pass


g = ConnCtx()
