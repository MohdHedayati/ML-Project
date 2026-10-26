"""
Deprecated alias module.
Re-exports AdaBN architecture from models.adabn.model.
"""

from models.adabn.model import AdaBN as ACO, AdaBN, AdaBNModel, TrafficSignNet

__all__ = ['ACO', 'AdaBN', 'AdaBNModel', 'TrafficSignNet']
