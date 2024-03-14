from .BaseFastFiz import BaseFastFiz
from .BaseRLFastFiz import BaseRLFastFiz

from gymnasium import register

register(
    id='BaseFastFiz-v0',
    entry_point=BaseFastFiz,
    disable_env_checker=True
)

register(
    id='BaseRLFastFiz-v0',
    entry_point=BaseRLFastFiz,
    disable_env_checker=True
)
