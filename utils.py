# -*- coding: utf-8 -*-

"""
momo.utils
~~~~~~~~~~~~~~~

This module provides common utils, as a standalone util module,
it shouldn't import any other modules from package
"""

import logging
import sys
import os
import re
import hashlib
import time
import uuid
import click
import decimal
import datetime
import numbers
import geohash
import random
import string
import inspect
import urlparse
import subprocess

import sqlalchemy as sa

from types import StringTypes
from collections import defaultdict
from thriftpy.thrift import TType
from pymysql.converters import convert_datetime
from gunicorn_thrift.utils import load_obj as origial_load_obj

from .consts import TO_SHIED_COMMON_PARAMETERS


# used for mobile verify
MOBILE_RE = re.compile('^1\d{10}$')
ALL_PHONE_RE = re.compile('^1\d{10}$|^[2-9]\d{6,7}(-\d{1,4})?$'
                          '|^6[1-9]{2,5}$')

# used for email verify
EMAIL_RE = re.compile('\w+([-+.]\w+)*@\w+([-.]\w+)*\.\w+([-.]\w+)*')

# used for number translate
UNIT_MAP = {u'十': 10, u'百': 100, u'千': 1000, u'万': 10000, u'亿': 100000000}
NUM_MAP = u"零一二三四五六七八九"

# used for capturing timeout error
TServerTimeoutSentryProject = None

LIB_DIR_PATH = os.path.dirname(os.path.abspath(__file__))

TYPE_MAP = {
    TType.BOOL: bool,
    TType.DOUBLE: float,
    TType.I16: (int, long),
    TType.I32: (int, long),
    TType.I64: (int, long),
    TType.STRING: (str, unicode),
}

DEPRECATION_PATTERN = 'option %r is deprecating, please use %r instead'


class GeoHashConversionException(ValueError):
    pass


def is_mobile(s):
    """
    Check whether a string is mobile number.

    :param s: mobile string to be checked
    """
    return MOBILE_RE.match(s) is not None


def is_phone(s):
    """
    Check whether a string is a fixed phone format.

    :param s: phone string to be checked
    """

    return ALL_PHONE_RE.match(s) is not None


def is_email(s):
    """
    Check whether a string is email address.

    :param s: email string to be checked
    """
    return EMAIL_RE.match(s) is not None


def datetime2utc(dt):
    """
    Sends a :class:`datetime.datetime` object.
    Returns :class:`int` time value.

    :param dt: :class:`datetime` object
    """
    return int(time.mktime(dt.timetuple()))


def utc2datetime(t):
    """
    Sends a :class:`float` time value.
    Returns :class:`datetime.datetime` object.

    :param t: :class:`float` time value.
    """
    return datetime.datetime.fromtimestamp(t)


def strptime2date(date_str):
    return datetime.datetime.strptime(date_str, '%Y-%m-%d').date()


def chinese2num(s):
    n = value = unit = flag = 0
    for i in range(len(s)):
        if s[i] in NUM_MAP:
            value = NUM_MAP.index(s[i])
            n += value * unit / 10 if flag and i == len(s) - 1 else value
            flag = 0
        elif s[i] in UNIT_MAP:
            unit = flag = UNIT_MAP[s[i]]
            if unit >= 10000:
                return n * unit + chinese2num(s[i + 1:])
            n += (unit - 1) * value if value else unit
        else:
            raise ValueError
    return n


def md5(s):
    """
    Returns :class:`str` object after md5 encrypt.

    :param s: :class:`str`, the str to md5 encrypt.
    """
    return hashlib.new('md5', s).hexdigest()


def sha1(s):
    """
    Returns :class:`str` object after sha1 encrypt.

    :param s: :class:`str`, the str to sha1 encrypt.
    """
    return hashlib.new('sha1', s).hexdigest()


def get_unique_string(length=31):
    uid = uuid.uuid1()
    full = md5(uid.hex)
    if length > 0:
        return full[:length]
    else:
        return full


def _serialize(obj, tobj, fields=None, excluded=()):
    fields = fields or tobj.__dict__
    types = {item[1]: TYPE_MAP[item[0]] for item in tobj.thrift_spec.values()
             if item[0] in TYPE_MAP}

    for k in fields:
        if k in excluded:
            continue

        if hasattr(obj, "_cached_property"):
            obj._cached_property = {}

        attr = getattr(obj, k)

        if hasattr(obj, "__table__"):
            column = obj.__table__.columns.get(k)

            if column is not None and attr is None and \
                    column.default is not None and \
                    column.default.is_scalar:
                attr = column.default.arg

            if column is not None and \
                    isinstance(attr, (str, unicode)) and \
                    isinstance(column.type, (sa.DateTime, sa.Date)):
                attr = convert_datetime(attr)

        if isinstance(attr, datetime.datetime):
            attr = datetime2utc(attr)
        elif isinstance(attr, decimal.Decimal):
            attr = float(attr)
        elif isinstance(attr, (datetime.date, datetime.time)):
            attr = unicode(attr)
        elif attr and k in types and not isinstance(attr, types[k]):
            _t = types[k]
            if isinstance(_t, tuple):
                _t = _t[0]
            attr = _t(attr)

        setattr(tobj, k, attr)


def serialize_with_excluded(obj, ttype=None, excluded=()):
    """
    :param obj: sqlalchemy model instance to be serialized
    ``__thrift__`` class attribute in the model is assumed
    :param ttype: Thrift struct type, must provide if ``obj`` has
    no ``__thrift__`` attribute
    :param excluded: fields which would not serialize from obj and just
     set to be None as for thrift's strict type
    :return: serialized result
    """
    if hasattr(obj, '__thrift__'):
        tobj = obj.__thrift__()
    else:
        tobj = ttype()

    fields = tobj.__dict__
    _serialize(obj, tobj, fields, excluded=excluded)
    return tobj


def serialize_to_ttype(obj, ttype, fields=None):
    tobj = ttype()
    _serialize(obj, tobj, fields)
    return tobj


def geo_point_to_geo_hash(latitude, longitude):
    if not latitude and not longitude:
        return 0
    hash_str = geohash.encode(latitude, longitude, 8)
    return int(hash_str.encode("hex"), 16) - 2 ** 63


def geo_hash_to_geo_point(hash_int):
    if not isinstance(hash_int, numbers.Number):
        raise GeoHashConversionException("`hash_int` should be a number")
    if hash_int >= 0:
        raise GeoHashConversionException("`hash_int` should be negative")
    hashcode = hex(int(hash_int + 2 ** 63))[2:].decode("hex")
    return geohash.decode(hashcode)


def clear_cache(redis_client, service_name, api_name, params):
    key = u"cache:{0}:{1}:{2}".format(service_name, api_name,
                                      u":".join(params or []))
    redis_client.delete(key)


def random_number(length):
    assert length > 0
    return random.randrange(10 ** (length - 1), 10 ** length)


def random_string(length):
    assert length > 0
    return ''.join(random.choice(string.letters + string.digits)
                   for x in range(length))


def median(nums):
    nums.sort()
    length = len(nums)
    if length % 2 == 0:
        return (nums[length / 2] + nums[length / 2 - 1]) / 2
    else:
        return nums[length / 2]


def in_maintenance_hour():
    from .conf import settings
    return datetime.datetime.now().hour in settings.MAINTENANCE_HOURS


def monetize(decimal_num):
    if isinstance(decimal_num, decimal.Decimal):
        return decimal_num.quantize(decimal.Decimal('0.01'))
    else:
        return decimal.Decimal(decimal_num).quantize(
            decimal.Decimal('0.01'))


def check_params(*args):
    def wrapper(f):
        arg_names, varargs, varkw, defaults = inspect.getargspec(f)
        checkers = dict(args)
        args_checkers = [checkers.get(i) for i in arg_names]

        def process(*args, **kwds):
            for arg, checker in zip(args, args_checkers):
                checker and checker(arg)

            for k, v in kwds.iteritems():
                checker = checkers.get(k)
                checker and checker(v)

            return f(*args, **kwds)

        return process

    return wrapper


def igroup(iterable, item_per_grp):
    more = [True]
    iterable = iter(iterable)

    def next_group():
        for _ in xrange(item_per_grp):
            try:
                yield next(iterable)
            except StopIteration:
                more[0] = False  # we don't have nonlocal in 2.7 ;(
                break

    while more[0]:
        yield next_group()


def safe_int(num):
    """Transform a string into int type with safety."""
    # this function is mainly designed for elasticsearch
    # since there are lots of bad int types
    num = num.strip()
    if '.' in num:
        # when the num is something like: `88.`
        return int(float(num))
    return int(num)


def dsn2dict(dsn):
    url = urlparse.urlparse(dsn)
    return {
        "host": url.hostname,
        "port": url.port
    }


def cast_bytes(s, encoding='utf8', errors='strict'):
    """cast unicode or bytes to bytes"""
    if isinstance(s, bytes):
        return s
    elif isinstance(s, unicode):
        return s.encode(encoding, errors)
    else:
        raise TypeError("Expected unicode or bytes, got %r" % s)


def cast_unicode(s, encoding='utf8', errors='strict'):
    """cast bytes or unicode to unicode"""
    if isinstance(s, bytes):
        return s.decode(encoding, errors)
    elif isinstance(s, unicode):
        return s
    else:
        raise TypeError("Expected unicode or bytes, got %r" % s)


# give short 'b' alias for cast_bytes, so that we can use fake b('stuff')
# to simulate b'stuff'
b = asbytes = cast_bytes
u = cast_unicode


class Version(object):
    """Abstract base class for version numbering classes.  Just provides
    constructor (__init__) and reproducer (__repr__), because those
    seem to be the same for all version numbering classes.
    """

    def __init__(self, vstring=None):
        if vstring:
            self.parse(vstring)

    def parse(self, vstring):
        raise NotImplementedError

    def __repr__(self):
        return "%s ('%s')" % (self.__class__.__name__, str(self))


class StrictVersion(Version):
    """Version numbering for anal retentives and software idealists.
    Implements the standard interface for version number classes as
    described above.  A version number consists of two or three
    dot-separated numeric components, with an optional "pre-release" tag
    on the end.  The pre-release tag consists of the letter 'a' or 'b'
    followed by a number.  If the numeric components of two version
    numbers are equal, then one with a pre-release tag will always
    be deemed earlier (lesser) than one without.

    The following are valid version numbers (shown in the order that
    would be obtained by sorting according to the supplied cmp function):

        0.4       0.4.0  (these two are equivalent)
        0.4.1
        0.5a1
        0.5b3
        0.5
        0.9.6
        1.0
        1.0.4a3
        1.0.4b1
        1.0.4

    The following are examples of invalid version numbers:

        1
        2.7.2.2
        1.3.a4
        1.3pl1
        1.3c4

    The rationale for this version numbering system will be explained
    in the distutils documentation.
    """

    version_re = re.compile(r'^(\d+) \. (\d+) (\. (\d+))? ([ab](\d+))?$',
                            re.VERBOSE)

    def parse(self, vstring):
        match = self.version_re.match(vstring)
        if not match:
            raise ValueError("invalid version number '%s'" % vstring)

        (major, minor, patch, prerelease, prerelease_num) = \
            match.group(1, 2, 4, 5, 6)

        if patch:
            self.version = tuple(map(string.atoi, [major, minor, patch]))
        else:
            self.version = tuple(map(string.atoi, [major, minor]) + [0])

        if prerelease:
            self.prerelease = (prerelease[0], string.atoi(prerelease_num))
        else:
            self.prerelease = None

    def __str__(self):

        if self.version[2] == 0:
            vstring = string.join(map(str, self.version[0:2]), '.')
        else:
            vstring = string.join(map(str, self.version), '.')

        if self.prerelease:
            vstring = vstring + self.prerelease[0] + str(self.prerelease[1])

        return vstring

    def __cmp__(self, other):
        if isinstance(other, StringTypes):
            other = StrictVersion(other)

        compare = cmp(self.version, other.version)
        if compare == 0:  # have to compare prerelease

            # case 1: neither has prerelease; they're equal
            # case 2: self has prerelease, other doesn't; other is greater
            # case 3: self doesn't have prerelease, other does: self is greater
            # case 4: both have prerelease: must compare them!

            if not self.prerelease and not other.prerelease:
                return 0
            elif self.prerelease and not other.prerelease:
                return -1
            elif not self.prerelease and other.prerelease:
                return 1
            elif self.prerelease and other.prerelease:
                return cmp(self.prerelease, other.prerelease)

        else:  # numeric versions don't match --
            return compare  # prerelease stuff doesn't matter


def filter_apis_to_shield(grpc, service_name):
    '''
    get the offset of parameter whose name containe 'password'
    '''
    apis_to_shield = defaultdict(set)
    service = getattr(grpc, service_name)
    for api in service.grpc_services:
        grpc_spec = getattr((getattr(service, api + "_args", None)),
                              "grpc_spec", dict())
        for offset, argu_name in grpc_spec.iteritems():
            for api_name_seg in TO_SHIED_COMMON_PARAMETERS:
                if api_name_seg in argu_name[1]:
                    apis_to_shield[api].add(offset - 1)
    return apis_to_shield


class EmptyValue(object):
    """represent a value that is empty, see :meth:`is_empty` to see what
    empty means here
    """
    def __init__(self, val):
        if not self.is_empty(val):
            raise ValueError('error invalid empty value: %r' % val)
        self.val = val

    @staticmethod
    def is_empty(val):
        """
        empty value means the following stuff:

            ``None``, ``[]``, ``()``, ``{}``, ``''``
        """
        return val not in (False, 0) and not val


class CachedProperty(object):
    """Decorator like python built-in ``property``, but it results only
    once call, set the result into decorated instance's ``__dict__`` as
    a static property.
    """
    def __init__(self, fn):
        self.fn = fn
        self.__doc__ = fn.__doc__

    def __get__(self, inst, cls):
        if inst is None:
            return self
        val = inst.__dict__[self.fn.__name__] = self.fn(inst)
        return val


# alias
cached_property = CachedProperty


class LazyLoadProperty(object):
    """Lazy load object from module.

    ``import_path`` like this: "tracker:tracker".
    """

    def __init__(self, import_path):
        self._import_path = import_path
        self._obj = None

    def __get__(self, obj, obj_type):
        if self._obj is None:
            self._obj = load_obj(self._import_path)
        return self._obj


lazy_load_property = LazyLoadProperty


def warn_deprecation(logger_or_name, old, new, lvl=logging.WARNING,
                     pat=DEPRECATION_PATTERN):
    if isinstance(logger_or_name, logging.Logger):
        logger = logger_or_name
    else:
        logger = logging.getLogger(logger_or_name)
    logger.log(lvl, pat, old, new)


class clickparams:
    """contains click param types"""


class ListParamType(click.ParamType):
    """ List param type for click Commands """
    name = 'list'

    IGNORE_STARTER = '['
    IGNORE_ENDER = ']'
    SPLITER = ','

    def convert(self, value, param, ctx):
        try:
            value = value.strip()
            if value[0] == self.IGNORE_STARTER:
                value.lstrip(self.IGNORE_STARTER)
            if value[-1] == self.IGNORE_ENDER:
                value.rstrip(self.IGNORE_ENDER)

            elements = value.split(self.SPLITER)
            return [e.strip() for e in elements]
        except BaseException:
            self.fail('%s is not a valid list' % value, param, ctx)


clickparams.LIST = ListParamType()  # alias


class SpacedKeyValueDict(click.ParamType):
    """take a spaced key-value string and convert it into a dict
    valid input looks like: ``"app=biz.member cluster=web"``
    """
    name = 'spaced_dict'

    def convert(self, value, param, ctx):
        if value is None:
            return
        if isinstance(value, SpacedKeyValueDict):
            return value
        try:
            res = {}
            val_list = value.strip().split(' ')
            for v in val_list:
                key, val = map(str.strip, v.split('='))
                res[key] = val
        except ValueError:
            self.fail('invalid format', param, ctx)
        else:
            return res


clickparams.SPACED_DICT = SpacedKeyValueDict()


def load_obj(import_path):
    """ lood obj by python path string. Add paramter check """
    if not import_path:
        return None
    return origial_load_obj(import_path)


def sqlalchemy_init_new_pool(session):
    if callable(session):
        session = session()
    for e in session.engines.itervalues():
        e.pool = e.pool.recreate()


def gen_task_hash(conn, task_name, task_args):
    """
    generate unique hash string for a single mysql write task

    .. seealso:: :func:`.tx_task.gen_task_hash` (async mysql
    transaction task hash generation)

    :param conn: a connectable of sqlalchemy
    :param func task_func: the actual running task function
    :param tuple task_args: arguments passed to task function
    """
    entropy = "{0} -> {1}".format(task_name, task_args)
    return md5(entropy)


def check_tool(name, msg=None):
    try:
        subprocess.check_call(['which', name], stdout=subprocess.PIPE)
    except subprocess.CalledProcessError:
        raise RuntimeError(msg or "Install dot first[brew install %s]" % name)


def import_obj(obj_path, hard=False):
    """
    import_obj imports an object by uri, example::

        >>> import_obj("module:main")
        <function main at x>

    :param obj_path: a string represents the object uri.
    ;param hard: a boolean value indicates whether to raise an exception on
                import failures.
    """
    try:
        # ``__import__`` of Python 2.x could not resolve unicode, so we need
        # to ensure the type of ``module`` and ``obj`` is native str.
        module, obj = str(obj_path).rsplit(':', 1)
        m = __import__(module, globals(), locals(), [obj], 0)
        return getattr(m, obj)
    except (ValueError, AttributeError, ImportError):
        if hard:
            raise


class EnvvarReader(dict):
    __uninitialized = object()

    def __init__(self, *envvars):
        for ev in envvars:
            ev_norm = self._normalize(ev)
            if ev != ev_norm:
                raise ValueError(
                    'envvar name should be UPPERCASED: {!r}'.format(ev))
            dict.__setitem__(self, ev_norm, self.__uninitialized)

    @cached_property
    def configured(self):
        return any([self[k] for k in self])

    def _normalize(self, varname):
        return varname.strip().upper()

    def refresh(self, key):
        key = self._normalize(key)
        dict.__setitem__(self, key, self.__uninitialized)

    def __getitem__(self, key):
        key = self._normalize(key)
        value = dict.__getitem__(self, key)
        if value is self.__uninitialized:
            value = os.getenv(key, None)
            dict.__setitem__(self, key, value)
        return value

    __getattr__ = __getitem__

    def get_int(self, key):
        value = self.__getitem__(key)
        if value is not None:
            return int(value)

    def set_int(self, key, value):
        value = str(value)
        if value.isdigit():
            key = self._normalize(key)
            os.environ[key] = value
            self.refresh(key)

    def __setitem__(self, key, value):
        raise RuntimeError('setting envvar not supported')

    __setattr__ = __setitem__


try:
    from setproctitle import setproctitle as _setproctitle

    def setproctitle(title):
        _setproctitle("vos: %s" % title)
except ImportError:
    def setproctitle(title):
        return


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(
                Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def get_cpu_count():
    if 'bsd' in sys.platform or sys.platform == 'darwin':
        cpu_count = 4
    else:
        cpu_count = os.sysconf('SC_NPROCESSORS_ONLN')
    return cpu_count


def function_key_generator(namespace, func, to_str=True, **kwargs):

    args = inspect.getargspec(func)
    has_self = args[0] and args[0][0] in ("self", "cls")

    def unicode_to_str(u):
        if to_str and isinstance(u, unicode):
            u = u.encode("utf-8")
        return u

    def pair_to_str(pair):
        k, v = pair
        return k, unicode_to_str(v)

    def generate_key(*args, **kw):
        if has_self:
            args = args[1:]
        args = tuple(map(unicode_to_str, args))
        tuples = sorted(map(pair_to_str, kw.iteritems()))
        return "{}|{}{}".format(namespace, args, tuples)
