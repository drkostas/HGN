from typing import List
import logging
from termcolor import colored


class ColorLog(object):
    __slots__ = ('_log', '_color', '_attrs', 'debug', 'info', 'warn', 'warning',
                    'error', 'exception', 'critical')

    _log: logging.Logger
    _color: str
    _attrs: List

    def __init__(self, logger: logging.Logger, color: str = None, attrs: List = None):
        self._log = logger
        self._color = color if color else 'white'
        self._attrs = attrs if attrs else ['bold']

    def __getattr__(self, name):
        if name in ['debug', 'info', 'warn', 'warning',
                    'error', 'exception', 'critical']:
            return lambda s, *args: getattr(self._log, name)(
                colored(s, color=self._color, attrs=self._attrs), *args)

        return getattr(self._log, name)

log = ColorLog(logging.getLogger(__name__))