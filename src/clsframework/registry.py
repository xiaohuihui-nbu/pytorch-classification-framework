import importlib

REGISTRIES = {"model": {}, "dataset": {}, "loss": {}}


def register(kind, name):
    """Decorator for explicitly installed Python plugins. See docs/USAGE.md."""

    def decorator(factory):
        if name in REGISTRIES[kind]:
            raise ValueError(f"Duplicate {kind} registration: {name}")
        REGISTRIES[kind][name] = factory
        return factory

    return decorator


def load_plugins(names):
    for name in names:
        importlib.import_module(name)


def lookup(kind, name):
    if name not in REGISTRIES[kind]:
        raise ValueError(f"Unknown {kind}: {name}; choices={sorted(REGISTRIES[kind])}")
    return REGISTRIES[kind][name]


MODEL_CATALOG = [
    "resnet18",
    "resnet50",
    "mobilenetv3_small_100",
    "efficientnet_b0",
    "convnext_tiny",
    "vit_tiny_patch16_224",
    "deit_tiny_patch16_224",
    "swin_tiny_patch4_window7_224",
]
DATASET_CATALOG = ["mnist", "fashionmnist", "cifar10", "cifar100", "pets", "flowers102"]
LOSS_CATALOG = ["cross_entropy", "soft_target_ce", "bce_with_logits", "softmax_focal", "sigmoid_focal"]
