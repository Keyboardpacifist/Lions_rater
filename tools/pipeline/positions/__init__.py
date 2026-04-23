"""
Position config registry.

To add a new position:
1. Copy _template.py to {position}.py
2. Fill in the config
3. Add it to POSITIONS below
"""
from .de import DE_CONFIG
from .dt import DT_CONFIG
from .lb import LB_CONFIG
from .p import P_CONFIG
from .qb import QB_CONFIG
from .rb import RB_CONFIG
from .wr import WR_CONFIG

POSITIONS = {
    "qb": QB_CONFIG,
    "wr": WR_CONFIG,
    "rb": RB_CONFIG,
    "de": DE_CONFIG,
    "dt": DT_CONFIG,
    "lb": LB_CONFIG,
    "p": P_CONFIG,
}
