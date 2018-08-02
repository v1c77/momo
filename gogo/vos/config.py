# -*- coding: utf-8 -*-

app_config = None


def load_app_config(raise_exc=False):
    """Load app config in lazy mode

        1. If current running app type is grpc or not set, load grpc
            app config.
        2. If current running app type is set wrong, raise exception.
    """

    global app_config

    if app_config is None:
        from gogo.vos import env

        # TODO(vici) add grpcAppConfig class
        if env.is_grpc_app():
            app_config = GrpcAppConfig().load(raise_exc=raise_exc)

        else:
            # TODO(vici) can add more type app like thrift wsgi
            # TODO(vici) raise app type error
            pass

    return app_config
