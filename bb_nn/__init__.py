# bb_nn: Neural network components for builderbrain
# Contains dual-rail architecture, program adapters, and fusion mechanisms

from . import dual_rail
from . import program_adapters
from . import fusion_gates
from . import composition_blocks

__all__ = ["dual_rail", "program_adapters", "fusion_gates", "composition_blocks"]
