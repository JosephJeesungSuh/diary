import inspect, asyncio, time, random, logging
from typing import Callable, Collection, Optional, Type

def retry_with_exponential_backoff(
    initial_delay: float = 1.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    max_retries: int = 10,
    no_retry_on: Optional[Collection[Type[Exception]]] = None,
) -> Callable[[Callable], Callable]:

    def decorator(func: Callable):
        
        if inspect.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                delay, retries = initial_delay, 0
                while True:
                    try:
                        return await func(*args, **kwargs)
                    except Exception as e:
                        if no_retry_on and type(e) in no_retry_on:
                            raise
                        if retries >= max_retries:
                            raise
                        await asyncio.sleep(delay)
                        delay *= exponential_base * (1 + jitter * random.random())
                        retries += 1
                        logging.warning(f"Async retry {retries}/{max_retries} after: {e}")
            return async_wrapper
        
        else:
            def sync_wrapper(*args, **kwargs):
                delay, retries = initial_delay, 0
                while True:
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        if no_retry_on and type(e) in no_retry_on:
                            raise
                        if retries >= max_retries:
                            raise
                        time.sleep(delay)
                        delay *= exponential_base * (1 + jitter * random.random())
                        retries += 1
                        logging.warning(f"Retry {retries}/{max_retries} after: {e}")
            return sync_wrapper
        
    return decorator
