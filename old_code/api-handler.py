import time
import logging
import tweepy

fmt = '%(asctime)s : %(module)s : %(levelname)s : %(message)s'
logging.basicConfig(format=fmt, level=logging.WARNING)
logger = logging.getLogger(__name__)

class AuthPoolAPI(object):
    def __init__(self, **kwargs):
        try:
            auths = kwargs.pop('auths')
            assert len(auths) > 0
        except KeyError:
            raise ValueError("Must provide auths")

        rate_limit_sleep = kwargs.pop('rate_limit_sleep', 15 * 60)
        rate_limit_retries = kwargs.pop('rate_limit_retries', 3)

        super(AuthPoolAPI, self).__init__(**kwargs)

        self.auths = auths
        self.rate_limit_sleep = rate_limit_sleep
        self.rate_limit_retries = rate_limit_retries
        self.apis = [tweepy.API(auth) for auth in self.auths]

    def __getattr__(self, name):
        avail = [
            hasattr(x, name) and hasattr(getattr(x, name), '__call__')
            for x in self.apis
        ]

        if not all(avail):
            raise AttributeError(name)

        # this function will proxy for the tweepy methods
        # "iself" to avoid confusing shadowing of the binding
        def func(iself, *args, **kwargs):
            retry_cnt = 0

            while True:
                for api in iself.apis:
                    method = getattr(api, name)

                    try:
                        return method(*args, **kwargs)
                    except tweepy.RateLimitError:
                        continue # try the next API object

                # we've tried all API objects and been rate
                # limited on all of them
                if retry_cnt < iself.rate_limit_retries:
                    msg = 'Rate limited on try {0}; sleeping {1}'
                    msg = msg.format(retry_cnt, iself.rate_limit_sleep)
                    logger.warning(msg)

                    time.sleep(iself.rate_limit_sleep)

                    retry_cnt += 1
                else:
                    msg = 'Rate limited in call to {0}'.format(name)
                    raise tweepy.RateLimitError(msg)

        # tweepy uses attributes to control cursors and pagination, so
        # we need to set them
        methods = [getattr(x, name) for x in self.apis]

        attribs = {}
        for k, v in vars(methods[0]).items():
            if all([k in vars(x).keys() for x in methods]):
                if all([vars(x)[k] == v for x in methods]):
                    setattr(func, k, v)

        setattr(self, name, func.__get__(self))

        return getattr(self, name)