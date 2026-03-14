# -*- coding: utf-8 -*-
import asyncio
import logging
import os
import sys

import coloredlogs
from loguru import logger

if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logger.remove()
logger.add(sys.stderr, level=logging.ERROR)


class InterceptHandler(logging.Handler):
    """Intercept standard logging messages toward your Loguru sinks.
    Code was taken from https://github.com/Delgan/loguru#entirely-compatible-with-standard-logging"""

    loggers = {}

    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = sys._getframe(2), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        if record.name not in self.loggers:
            self.loggers[record.name] = logger.bind(name=record.name)
        self.loggers[record.name].opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


coloredlogs.DEFAULT_LEVEL_STYLES = {
    **coloredlogs.DEFAULT_LEVEL_STYLES,
    'critical': {'background': 'red'},
    'debug': coloredlogs.DEFAULT_LEVEL_STYLES['info'],
}

log_level = logging.DEBUG if os.environ.get('PRODUCTION', False) == 'DEBUG' else logging.INFO
if isinstance(log_level, str):
    log_level = logging.INFO

format_string = '%(asctime)s | %(name)s | %(levelname)s | %(message)s'

coloredlogs.install(stream=sys.stdout, level=log_level, fmt=format_string)

logging.basicConfig(level=log_level, format=format_string)
logging.getLogger().addHandler(InterceptHandler(level=log_level))

logging.getLogger('github.Requester').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('multipart').setLevel(logging.WARNING)
logging.getLogger('oauth2client').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.INFO)
logging.getLogger('pdfminer').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('websockets').setLevel(logging.WARNING)
logging.getLogger('werkzeug').setLevel(logging.WARNING)
