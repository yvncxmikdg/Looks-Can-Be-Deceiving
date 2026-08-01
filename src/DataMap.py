from collections import defaultdict
from copy import deepcopy

import yaml

class Labels2IdxMap:
    def __init__(self, labels2idx_map, background_class_idx=None):
        self._background_class_idx = background_class_idx
        self._labels_2_idx = labels2idx_map
        self._idx_2_labels = defaultdict(list)
        for label in self._labels_2_idx.keys():
            self._idx_2_labels[self._labels_2_idx[label]].append(label)
    def getBackgroundClassIdx(self):
        return self._background_class_idx
    def getBackgroundClass(self):
        return self._idx_2_labels[self._background_class_idx]
    def getIndex(self, label):
        return self._labels_2_idx[label]
    def getLabels(self, index):
        return self._idx_2_labels[index]
    def __len__(self):
        return len(self._idx_2_labels)
    def getAllLabels(self):
        return list(self._labels_2_idx.keys())
    def __getitem__(self, x, objtype=None):
        try:
            return self._labels_2_idx[x]
        except KeyError:
            pass
        try:
            return self._idx_2_labels[x]
        except KeyError:
            pass
        return None
    def __eq__(self, other):
        if not isinstance(other, Labels2IdxMap):
            return False
        if other._labels_2_idx != self._labels_2_idx:
            return False
        if other._idx_2_labels != self._idx_2_labels:
            return False
        if other._background_class_idx != self._background_class_idx:
            return False
        return True
    def __ne__(self, other):
        return not self.__eq__(other)

class DefaultLabel2IdxMap(Labels2IdxMap):
    def __init__(self, default_val):
        super().__init__({}, None)
        self._default_val = default_val
    def getBackgroundClassIdx(self):
        return 0
    def getBackgroundClass(self):
        return []
    def getIndex(self, label):
        return self._default_val
    def getLabels(self, index):
        return []
    def __len__(self):
        return 2
    def getAllLabels(self):
        return []
    def __getitem__(self, x, objtype=None):
        return None
    def __eq__(self, other):
        if not isinstance(other, DefaultLabel2IdxMap):
            return False
        if other._default_val != self._default_val:
            return False
        return super().__eq__(other)

class ColorMap:
    def __init__(self, label2Color_map, label2idx_map, color_format=None):
        self._label_2_color = label2Color_map
        self._label_2_idx = label2idx_map
        self._idx_2_labels = defaultdict(list)
        for label in self._label_2_idx.keys():
            self._idx_2_labels[self._label_2_idx[label]].append(label)
        self._color_format = ["red", "green", "blue", "alpha"] if color_format is None else color_format
        self._get_color_format_func = lambda x: [x[f] for f in self._color_format]

    def _get_label_to_use(self, label=None, idx=None):
        label_to_use = label
        if not idx is None:
            labels = self._idx_2_labels[idx]
            if len(labels) > 1:
                raise ValueError("Attempting to plot the color of index " + str(idx) + " which has conflicting labels: " + str(labels))
            label_to_use = labels[0]
        return label_to_use

    def getColorFormatted(self, label=None, idx=None):
        return self._get_color_format_func(self.getColorDict(label, idx))

    def getColorDict(self, label=None, idx=None):
        return self._label_2_color[self._get_label_to_use(label, idx)]

    def getLabelsInIndexOrder(self, include_transparent=False):
        """Output-class labels ordered by their model index (i.e. damage-severity order).

        By default the transparent background class (alpha 0) is dropped, since the
        deliverables only ever plot the real classes; pass include_transparent=True to
        keep it. This lets callers take the class list and its order from the channel map
        instead of hardcoding them.
        """
        ordered = sorted(self._label_2_idx, key=lambda label: self._label_2_idx[label])
        if include_transparent:
            return ordered
        return [label for label in ordered if self._label_2_color.get(label, {}).get("alpha", 255)]

class Channel2IdxMap:
    def __init__(self, channel2idx_map):
        self._channel_2_idx = channel2idx_map
        self._idx_2_channel = {idx:channel for channel, idx in self._channel_2_idx.items()}

    def getChannel(self, idx):
        return self._idx_2_channel[idx]
    def getIdx(self, channel):
        return self._channel_2_idx[channel]
    def __len__(self):
        return len(self._channel_2_idx)
    def items(self):
        return self._channel_2_idx.items()
    def dict(self):
        return deepcopy(self._channel_2_idx)


def color_map_from_channels_parameters_file(channels_parameters_file):
    """Build a ColorMap from a model's channels parameters YAML.

    The channel map's ``model_class_2_color_map`` is the single source of truth for the
    categorical damage-class colors shown across the deliverables (inspection orthos, bar
    plots, and report/slide legends), keyed the same way as the model's output classes via
    ``output_class_2_idx_map``. See frontend/build_inspection_rda_ortho.py for the canonical
    inline construction this wraps.
    """
    with open(channels_parameters_file, encoding="utf-8") as stream:
        channel_maps = yaml.safe_load(stream)["channel_maps"]
    return ColorMap(channel_maps["model_class_2_color_map"],
                    channel_maps["output_class_2_idx_map"])
