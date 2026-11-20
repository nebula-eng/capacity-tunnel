"""capacity-tunnel: covert-tunnel detection via covert-channel capacity fingerprints.

Public surface:

``channel``   real UDP loopback covert channel (b32, timing, hardened)
``benign``    benign traffic generators (DoH browse, bulk, interactive)
``features``  per-flow feature extraction
``capacity``  covert-channel capacity estimator
``detector``  logistic-regression detector, ROC/AUC, cross-validation
"""

__version__ = "1.0.0"

from . import benign, capacity, channel, detector, features  # noqa: F401

__all__ = ["benign", "capacity", "channel", "detector", "features", "__version__"]
