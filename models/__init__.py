from .ast import ASTModel
from .projector import Projector

_backbone_class_map = {
    'ast': ASTModel,
}


def get_backbone_class(key):
    if key in _backbone_class_map:
        return _backbone_class_map[key]
    else:
        raise ValueError('Invalid backbone: {}'.format(key))