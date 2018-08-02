# -*- coding: utf-8 -*-

"""
Log
===

1. Provides a logging formmater with tags support.
2. Provides logging dict config for dev and prod env.
3. Provides a `context` function to feed tags.
4. Provides an alternatibe sql logger.
5. Automatic logging within api call context via signals.

Settings
--------
::

    SQLLOGGER_SWITCH        sql logger switch, default: lambda: True

Apis
----
"""
import six
import os
import contextlib
import inspect
import logging
import threading
import time
import functools

# TODO(vici) need mongoengine event  module import

from gogo.conf import settings
from gogo.consts import PLACE_HOLDER
from gogo.ctx import g
from gogo.exc import ServerRestartException
from gogo.vos.config import load_app_config
from gogo.vos import env
from utils import LIB_DIR_PATH
from gogo import signals

IGNORED_EXCEPTIONS = (ServerRestartException, )
_LOG_BLACKLIST_HUSKAR_KEY = 'VES_LOG_API_BLACKLIST'
_LOG_BLACKLIST = {'ping'}

#########
# Common Formatter
#########


def attach_logging_meta(func):
    """Decorator: add ip and func_name info to log.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        ip = g.get_conn_meta('client_addr')
        func_name = g.get_call_meta('method')
        with logging_meta(ip=ip, func_name=func_name):
            return func(*args, **kwargs)
    return wrapper


def obj2str(obj):
    """Try to convert an object to string format."""
    if isinstance(obj, six.unicode):
        return obj.encode('utf8')
    elif isinstance(obj, (str, int, float, bool)):
        return str(obj)
    return repr(obj)


def make_meta_dict(prefix, ins):
    """Add prefix to custom meta key.
    Value of dict items should be string; if not, ignore.
    """
    for key, val in ins.items():
        if not isinstance(val, basestring):
            ins.pop(key)
    fixed_dict = {"_".join((prefix, key)): ins[key] for key in ins}
    g.logging_meta.update(fixed_dict)


def logging_meta_before_called(**custom_metas):
    """Decorator: add custom_meta info into log meta before func execute.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            make_meta_dict("before-called", custom_metas)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def logging_meta_after_called(*meta_keys, **custom_metas):
    """Decorator: add custom_meta info into log meta after func execute.
    Value of meta_key in dict logging_meta will be func res or attr of res.
    Res of func should be basestring, int, bool or instance of class.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            res = func(*args, **kwargs)
            for meta_key in meta_keys:
                if isinstance(res, (basestring, int, float, bool)):
                    custom_metas[meta_key] = obj2str(res)
                else:
                    custom_metas[meta_key] = obj2str(
                        getattr(res, meta_key, None))
            make_meta_dict("after-called", custom_metas)
            return res
        return wrapper
    return decorator


@contextlib.contextmanager
def logging_meta(**kwargs):
    """Logging with key-value meta, e.g.
    ::

        with logging_meta(key=val) as ctx:
            # do logging
    """
    conflicts = {}
    meta = g.logging_meta

    # if has conflicts, record the original kvs
    for key in kwargs:
        if key in meta:
            conflicts[key] = meta[key]

    try:
        meta.update(kwargs)
        yield meta
    finally:
        # recovery the original conflict kvs
        for key in kwargs:
            meta.pop(key)
        meta.update(conflicts)


class ZeusFormatter(logging.Formatter):
    """
    Logging formmater with meta (kv pairs) support.

    Skeletal structure::

        [required_meta] extra_meta ## body

    Detail structure::

        datetime level name[pid]: [app_id rpc_id request_id] k => v .. ## msg

    """

    def _format(self, msg):
        # required meta
        app_id = load_app_config().app_id
        request_id = g.get_call_meta('request_id')
        rpc_id = g.get_call_meta('seq')
        required_meta = '[{0} {1} {2}]'.format(app_id, rpc_id, request_id)

        # extra_meta
        metas = [required_meta]
        if g.logging_meta:
            for item in g.logging_meta.iteritems():
                k, v = map(obj2str, item)
                extra_meta = '%s => %s' % (k, v)
                metas.append(extra_meta)

        # meta
        meta = ' '.join(metas)
        # body
        body = obj2str(msg)
        # join msg
        return ' ## '.join([meta, body])

    def format(self, record):
        record.msg = self._format(record.msg)
        ret = super(ZeusFormatter, self).format(record)
        if env.is_in_appos:
            ret = ret.replace("\n", "#012")
        return ret


#########
# SQL Logging
#########


class SQLLogger(object):
    """SQL logger listens sqlalchemy events on ``engines`` and emits
    logging after ``cursor`` executed::

        sql_logger = SQLLogger()
        sql_logger.register(engine)
    """
    # senstive db field list
    _path = os.path.join(LIB_DIR_PATH, 'static', 'dbsens')
    _content = open(_path).read()
    DB_SENSFIELDS = set(filter(None, _content.splitlines()))

    def __init__(self, name='sql_logger'):
        self.logger = logging.getLogger(name=name)
        env.require('gevent_patched')
        self.ctx = threading.local()
        self.ctx.disable_sql_logging = False
        self.ctx.sql_start_time = None
        self.switch = lambda: False
        self.signal = signals.mysql_signal_wrapper

    def use_switch(self, fn):
        """
        Usage::

            sql_logger.use_switch(lambda: True)  # on
        """
        self.switch = fn

    def register(self, engine):
        event.listen(engine, 'before_cursor_execute',
                     self.before_cursor_execute)
        event.listen(engine, 'after_cursor_execute',
                     self.after_cursor_execute)
        event.listen(engine, 'dbapi_error',
                     self.handle_dbapi_error)
        event.listen(engine, 'commit', self.on_commit)
        event.listen(engine, 'rollback', self.on_rollback)
        event.listen(engine, 'connect', self.on_connect)

    @contextlib.contextmanager
    def disable_sql_logging(self):
        """
        Usage example::

            with sql_logger.disable_sql_logging():
                session.add(User)
                session.commit()
        """
        self.ctx.disable_sql_logging = True
        try:
            yield self
        finally:
            self.ctx.disable_sql_logging = False

    def before_cursor_execute(self, conn, cursor, statement,
                              params, context, executemany):
        if not hasattr(self.ctx, 'disable_sql_logging'):
            self.ctx.disable_sql_logging = False
        self.ctx.sql_start_time = time.time()

    def after_cursor_execute(self, conn, cursor, statement,
                             params, context, executemany):
        self._do_logging(self._get_logging_ctx(conn), statement,
                         params=params, exc=None)

    def handle_dbapi_error(self, conn, cursor, statement,
                           params, context, exception):
        self._do_logging(self._get_logging_ctx(conn), statement,
                         params=params, exc=exception)

    def on_commit(self, conn):
        self._do_logging(self._get_logging_ctx(conn), 'COMMIT')

    def on_rollback(self, conn):
        self._do_logging(self._get_logging_ctx(conn), 'ROLLBACK')

    def on_connect(self, dbapi_conn, conn_record):
        self._do_logging(self._get_logging_ctx_by_dbapi_conn(dbapi_conn),
                         'Connection Open')

    def _get_logging_ctx(self, conn):
        if not conn.invalidated:  # make sure Connection.__connection exists
            dbapi_conn = conn.connection.connection
        else:
            dbapi_conn = None
            self.logger.warning('DB connection invalidated before logging')
        return self._get_logging_ctx_by_dbapi_conn(dbapi_conn,
                                                   driver=conn.engine.driver)

    def _get_logging_ctx_by_dbapi_conn(self, dbapi_conn, driver=None):
        default = {}.fromkeys(['db', 'host', 'port', 'autocommit'],
                              PLACE_HOLDER)

        if dbapi_conn is None:
            return default

        if driver is None:
            driver = dbapi_conn.__class__.__module__.split('.')[0]

        if driver == 'pymysql':
            return dict(db=dbapi_conn.db,
                        host=dbapi_conn.host,
                        port=dbapi_conn.port,
                        autocommit=dbapi_conn.get_autocommit())
        elif driver == 'psycopg2':
            dct = {}
            for item in dbapi_conn.dsn.split():
                lst = item.split('=')
                dct[lst[0]] = lst[1]
            return dict(db=dct.get('dbname', PLACE_HOLDER),
                        host=dct.get('host', PLACE_HOLDER),
                        port=dct.get('port', PLACE_HOLDER),
                        autocommit=dbapi_conn.autocommit)
        return default

    def _do_logging(self, ctx, statement, params=None, exc=None):
        # build trace ctx
        trace_ctx = signals.SignalContext()
        trace_ctx.start = None

        # time cost
        cost = None
        if getattr(self.ctx, 'sql_start_time', None) is not None:
            cost = time.time() - self.ctx.sql_start_time
            # need to record SQL exact start time for etrace
            trace_ctx.start = self.ctx.sql_start_time * 1000
            self.ctx.sql_start_time = None

        # build trace ctx continue
        trace_ctx.cost = cost and cost * 1000 or 0
        trace_ctx.db = ctx.get("db", "unknown")
        trace_ctx.exc = exc
        trace_ctx.statement = statement
        self.signal.after_mysql_execute.send(trace_ctx)

        if not hasattr(self, 'switch') or (
                callable(self.switch) and not self.switch()):
            # there's not a switch or there's a switch and it gives us `off`
            return

        if getattr(self.ctx, 'disable_sql_logging', False):
            # switch is `on` but logging is temporarily disabled for
            # current thread.
            return

        meta = {}
        if cost is None:
            meta['cost'] = PLACE_HOLDER
        else:
            meta['cost'] = '%.2fms' % (cost * 1000)

        # db instance
        meta['db'] = '{0}/{1}'.format(ctx.get('host', PLACE_HOLDER),
                                      ctx.get('db', PLACE_HOLDER))
        # autocommit
        if 'autocommit' in ctx and ctx['autocommit'] in (False, True):
            meta['ac'] = int(ctx['autocommit'])
        else:
            meta['ac'] = PLACE_HOLDER
        # result
        meta['res'] = exc or 'ok'
        # session id
        meta['session'] = getattr(self.ctx, 'current_session_id', PLACE_HOLDER)

        # filter senstive fields
        statement_list = statement.split()
        has_sens_field = set(statement_list) & SQLLogger.DB_SENSFIELDS
        statement_ = obj2str(' '.join(statement_list))

        if has_sens_field or params is None:
            params_ = PLACE_HOLDER
        else:
            params_ = obj2str(params)

        content = '{0} ## {1}'.format(statement_, params_)

        with logging_meta(**meta):
            if exc is None:
                if cost is None or cost < 1:
                    self.logger.info(content)
                elif 1 <= cost < 5:
                    self.logger.warning(content)
                else:
                    self.logger.error(content)
            else:
                self.logger.error(content)


sql_logger = None


def get_sql_logger():
    global sql_logger
    if sql_logger is None:
        sql_logger = SQLLogger()
        env.require('settings_updated')
        sql_logger.use_switch(settings.SQLLOGGER_SWITCH)
    return sql_logger


#########
# API Calling Logging
#########


def register_receivers():
    import signals
    signals.after_api_called_ok.connect(on_signal_after_api_called_ok)
    signals.after_api_called_t_exc.connect(on_signal_after_api_called_t_exc)
    signals.after_api_called_timeout.connect(
        on_signal_after_api_called_timeout)
    signals.after_api_called_unkwn_exc.connect(
        on_signal_after_api_called_unkwn_exc)

    signals.after_api_health_tested_bad.connect(
        on_signal_after_api_health_tested_bad)
    signals.after_api_health_locked.connect(
        on_signal_after_api_health_locked)
    signals.after_api_health_unlocked.connect(
        on_signal_after_api_health_unlocked)

    signals.alt_after_api_health_tested_bad.connect(
        on_signal_alt_after_api_health_tested_bad)
    signals.alt_after_api_health_locked.connect(
        on_signal_alt_after_api_health_locked)
    signals.alt_after_api_health_unlocked.connect(
        on_signal_alt_after_api_health_unlocked)

    wsgi_signal = signals.wsgi_signal_wrapper
    wsgi_signal.hard_timeout.connect(wsgi_called_hard_timeout)
    wsgi_signal.called_unkwn_exc.connect(wsgi_called_unkwn_exc)
    wsgi_signal.soft_timeout.connect(wsgi_called_soft_timeout)
    wsgi_signal.switch_off.connect(wsgi_called_switch_off)

    register_api_log_black_list()


def register_api_log_black_list():
    huskar_config = get_huskar_config()
    update_black_list(huskar_config.get(_LOG_BLACKLIST_HUSKAR_KEY))
    huskar_config.on_change(_LOG_BLACKLIST_HUSKAR_KEY)(update_black_list)


def update_black_list(value):
    if isinstance(value, list):
        global _LOG_BLACKLIST
        _LOG_BLACKLIST = set(value) | {'ping'}


def log_obscure_hook(pos, encrypt_func):
    """Decorator: add encrypt info to func as its attr.
    :param int pos: position of argument need encrypt, start from *1*.
    """
    def wrapper(func):
        args = inspect.getargspec(func)[0]
        eff_args = args[1:] if args[0] == "self" else args
        if not isinstance(pos, int) or not 0 < pos <= len(eff_args):
            raise ValueError("log_obscure_hook param error.")
        if not hasattr(func, "obscure_info"):
            func.obscure_info = []
        func.obscure_info.append((pos, encrypt_func))
        return func
    return wrapper


def encrypt_args(args, func):
    obscure_info = getattr(func, "obscure_info", None)
    if obscure_info:
        for pos, encrypt_func in obscure_info:
            try:
                pos -= 1
                args[pos] = encrypt_func(args[pos])
            except TypeError:
                pass
    return args


def sheild_args(ctx):
    """Dont logging args if the api function is senstive,
    e.g. `password` for `login`, set to ``-`` when
    logging."""
    log_args = encrypt_args(list(ctx.args), ctx.func)

    for offset in ctx.service.get('__apis_to_shield', {})\
            .get(ctx.func.func_name, set()):
        log_args[offset] = PLACE_HOLDER

    if not log_args:
        return '()'
    return '({0})'.format(','.join(repr(i) for i in log_args))


def build_sentry_data():
    end_time = time.time()

    call_func = g.get_call_meta("method")
    args = g.get_call_meta("args")

    dispatcher = args[0].__class__.__name__ if \
        isinstance(args, (list, tuple)) and \
        len(args) > 0 and \
        type(type(args[0])) is type else PLACE_HOLDER

    tags = {
        'api': call_func,
        'dispatcher': dispatcher
    }

    return dict(extra={
                'call': g.call_meta_data,
                'client': g.conn_ctx,
                'end': end_time
                },
                tags=tags)


@attach_logging_meta
def on_signal_after_api_called_ok(ctx):
    log_args_str = sheild_args(ctx)
    cost = ctx.cost

    # timed out warnings
    if cost > ctx.api_soft_timeout:
        ctx.logger.warning(
            'Soft Timed out! {0}{1} => {2}ms'.format(
                ctx.func.func_name, log_args_str, cost))
    # info logs
    if ctx.func.func_name not in _LOG_BLACKLIST:
        ctx.logger.info('{0}{1} response time {2}ms'.format(
            ctx.func.func_name, log_args_str, cost))


@attach_logging_meta
def on_signal_after_api_called_t_exc(ctx):
    cost = ctx.cost
    # Do not log restart exception in ping.
    if isinstance(ctx.exc, ServerRestartException):
        return

    log_args_str = sheild_args(ctx)
    msg = u'{0} {1} => {2}{3} response time {4}ms'.format(
        getattr(ctx.exc, 'error_code', ''), ctx.exc.message,
        ctx.func.func_name, log_args_str, cost)
    if isinstance(ctx.exc, ctx.service.user_exc):
        if ctx.exc.error_code in ctx.ignored_user_exception_codes:
            ctx.logger.info(msg)
        else:
            ctx.logger.warning(msg)
    else:
        ctx.logger.exception(msg, extra=build_sentry_data())


@attach_logging_meta
def on_signal_after_api_called_timeout(ctx):
    log_args_str = sheild_args(ctx)
    cost = ctx.cost
    msg = u'{0}:{3}ms => {1}{2}'.format(
        'Gevent Timeout!', ctx.handle_name,
        log_args_str, cost)
    ctx.logger.exception(msg, extra=build_sentry_data())


@attach_logging_meta
def on_signal_after_api_called_unkwn_exc(ctx):
    cost = ctx.cost
    log_args_str = sheild_args(ctx)
    if isinstance(ctx.exc, IGNORED_EXCEPTIONS):
        ctx.logger.warning('{0} => {1}{2} response time {3}ms'.format(
            repr(ctx.exc), ctx.handle_name, log_args_str, cost))
    else:
        ctx.logger.exception('{0} => {1}{2}'.format(
            repr(ctx.exc), ctx.handle_name, log_args_str),
            extra=build_sentry_data())


@attach_logging_meta
def on_signal_after_api_health_tested_bad(ctx):
    ctx.logger.error("API not healthy => {}".format(ctx.func_name))


@attach_logging_meta
def on_signal_after_api_health_locked(ctx):
    ctx.logger.error("API locked => {}".format(ctx.func_name))


@attach_logging_meta
def on_signal_after_api_health_unlocked(ctx):
    ctx.logger.info("API unlocked => {}".format(ctx.func_name))


@attach_logging_meta
def on_signal_alt_after_api_health_tested_bad(ctx):
    ctx.logger.error("Logic not healthy => {}".format(ctx.func_name))


@attach_logging_meta
def on_signal_alt_after_api_health_locked(ctx):
    ctx.logger.error("Logic locked => {}".format(ctx.func_name))


@attach_logging_meta
def on_signal_alt_after_api_health_unlocked(ctx):
    ctx.logger.info("Logic unlocked => {}".format(ctx.func_name))


@attach_logging_meta
def wsgi_called_hard_timeout(ctx):
    msg = u'Gevent timeout: {1}ms => {0}'.format(ctx.handle_name, ctx.cost)
    ctx.logger.exception(msg, extra=build_sentry_data())


@attach_logging_meta
def wsgi_called_soft_timeout(ctx):
    msg = u'Soft timeout: {1}ms => {0}'.format(ctx.handle_name, ctx.cost)
    ctx.logger.exception(msg, extra=build_sentry_data())


@attach_logging_meta
def wsgi_called_unkwn_exc(ctx):
    if isinstance(ctx.exc, IGNORED_EXCEPTIONS):
        ctx.logger.warning('{0} => {1} response time {2}ms'.format(
            repr(ctx.exc), ctx.handle_name, ctx.cost))
    else:
        ctx.logger.exception('{0} => {1}'.format(
            repr(ctx.exc), ctx.handle_name),
            extra=build_sentry_data())


@attach_logging_meta
def wsgi_called_switch_off(ctx):
    ctx.logger.warning('{0} => {1}.{2}'.format(ctx.exc, ctx.service_name,
                                               ctx.handle_name))
