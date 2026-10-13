
"""Volume datasets and RAW preparation tools.

Load the training datasets on demand so RAW preparation only requires NumPy.
"""

__all__ = ['CoreVolumeDataset', 'CoreVolumeDataset4x']


def __getattr__(name):
    if name in __all__:
        from dataset.core_volume import CoreVolumeDataset, CoreVolumeDataset4x
        globals().update(CoreVolumeDataset=CoreVolumeDataset,
                         CoreVolumeDataset4x=CoreVolumeDataset4x)
        return globals()[name]
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
