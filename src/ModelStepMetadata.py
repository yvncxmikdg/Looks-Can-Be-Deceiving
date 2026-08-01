import copy
from collections import defaultdict

class MetadataCollection:
    def __init__(self, functor=lambda:0, normalizable=False):
        self._fields = defaultdict(functor)
        self._normalizable = defaultdict(lambda: normalizable)
        self._normalization_value = 0
    def __getitem__(self, key):
        return self._fields[key]
    def __setitem__(self, key, value):
        self._fields[key] = value
    def keys(self):
        return self._fields.keys()
    def items(self):
        return self._fields.items()
    def values(self):
        return self._fields.values()
    def set_normalizable(self, key, normalizable=True):
        self._fields[key] = normalizable
    def is_normalizable(self, key):
        return self._fields[key]

class MultiScalars:
    def __init__(self, collection=None, normalizable=True):
        self.__scalar = MetadataCollection() if collection is None else collection
        self.__normalizable = normalizable
        self._normalization_value = 0
    def set_normalization_value(self, value):
        self._normalization_value = value
    def increment_normalization_value(self, increment_val=1):
        self._normalization_value += increment_val
    def get_normalization_value(self):
        return self._normalization_value
    def __setitem__(self, name, scalar):
        self.__scalar[name] = scalar
    def __getitem__(self, name):
        return self.__scalar[name]
    def get_scalar_names(self):
        return self.__scalar.keys()
    def as_dict(self):
        return copy.deepcopy(self.__scalar)
    def is_normalizable(self):
        return self.__normalizable

class AveragedMetadataCollection(MetadataCollection):
    def __init__(self, functor=lambda:[], normalizable=False):
        super().__init__(functor, normalizable)
    def get_averaged_value(self, key):
        return sum(float(v) for v in self._fields[key])
    def get_count_of_values(self, key):
        return len(self._fields[key])

class ModelStepMetadata:
    def __init__(self, step):
        self.__step = step
        self.reset()
    def get_step(self):
        return self.__step
    def reset(self):
        self.normalizations = MetadataCollection()
        self.images = MetadataCollection(lambda:None, False)
        self.scalar = MetadataCollection(normalizable=True)
        self.scalars = MetadataCollection(lambda:MultiScalars(collection=None, normalizable=True))
        self.quantiles = AveragedMetadataCollection()
