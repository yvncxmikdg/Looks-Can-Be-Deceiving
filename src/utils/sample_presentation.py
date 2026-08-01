import numpy as np

from modeling.utils.random_utils import reseed_distributed
from modeling.Spatial import Building, LabeledRoadLine, RoadLine

def distribution_proportion(arr, max_val=None, min_val=None):
    if min_val is None:
        min_val = np.min(arr)
    if max_val is None:
        max_val = np.max(arr)

    if max_val - min_val == 0:
        return np.array([1/len(arr)]*len(arr))

    return arr / np.sum(arr)

class SampleLocationPresentationStrategy:
    def __init__(self):
        pass
    def getSampleLocation(self, index):
        raise NotImplementedError("Function getSampleLocation must be implemented by a subclass.")
    def __len__(self):
        raise NotImplementedError("Function __len__ must be implemented by a subclass.")

#Below are the sample location presentation strategies to be used when the sample locations are known ahead of time
class PregeneratedSampleLocationPresentationStrategy(SampleLocationPresentationStrategy):
    def __init__(self):
        self._sample_locations = None
    def initialize_samples(self, sample_locations):
        self._sample_locations = sample_locations
    def getSampleLocation(self, index):
        return self._sample_locations[index]
    def __len__(self):
        return len(self._sample_locations)

class IndexSampleLocationPresentationStrategy(PregeneratedSampleLocationPresentationStrategy):
    pass

class WeightedSampleLocationPresentationStrategy(PregeneratedSampleLocationPresentationStrategy):
    """Presents training samples so the delivered class distribution is driven toward a target.

    At each draw every candidate sample is scored by how much presenting it would help close the
    current gap to the target distribution -- ``value(sample) = sum_k max(0, target_k - observed_k)
    * content_k(sample)`` where ``content_k`` is the sample's amount of class ``k`` (object count
    under the ``label`` monitor, pixel area/length under the ``pixel`` monitor) -- and a sample is
    drawn with probability proportional to its value. Classes below target pull toward samples that
    contain them, in proportion to how far below target they are; classes at or above target exert
    no pull. This makes the delivered distribution converge to the target regardless of how the pool
    is composed (imbalanced, clustered, or mixed-content), which a naive draw over samples/tiles
    would not.

    Args:
        length: number of samples this strategy reports via ``__len__`` (the epoch length).
        expected_class_balances: dict of class -> relative weight defining the target distribution;
            normalized to proportions internally (e.g. all-equal weights => a uniform target).
        sample_selection_smoothing: optional non-negative exploration floor added to every non-empty
            sample's value (default ``0.0`` = off). At ``0`` the sampler is purely deficit-driven and
            matches the target faithfully -- this is the recommended default. A positive value gives
            every non-empty sample a baseline chance of being drawn even when it holds no currently
            -deficient class (useful when you deliberately want extra sample diversity), but because
            the floor is added *per sample* it biases selection toward classes that have more sample
            tiles, degrading target-matching on imbalanced pools (an all-size-1 pool with many tiles
            of one class is the worst case). Keep it small relative to typical sample values. Empty
            samples always score zero and are never drawn regardless of this value. (This replaces
            the former ``sample_selection_smoothing_alpha``; the old alpha-beta smoothing -- and the
            ``_beta`` exponent -- are gone with the move to proportional per-sample weighting.)
        balance_monitor: ``"label"`` to measure class content by object count, or ``"pixel"`` to
            measure it by pixel area (buildings) / length (road lines).
    """
    def __init__(self, length, expected_class_balances, sample_selection_smoothing=0.0, balance_monitor="label"):
        super().__init__()
        self._length = length

        target_class_norm_denom = sum(expected_class_balances.values())
        self._sample_target_class_proportions = {k:v/target_class_norm_denom for k,v in expected_class_balances.items()}

        #Optional exploration floor added to each non-empty sample's value (default 0 = off); see the
        #class docstring for the tradeoff and _compute_sample_selection_probabilities for its use.
        self._sample_selection_smoothing = sample_selection_smoothing
        self._classes = list(expected_class_balances.keys())
        self._sample_class_observation_counts = {k:0 for k in expected_class_balances.keys()}
        #(num_samples, num_classes) matrix of per-class content, populated at initialize_samples.
        self._sample_content = None
        self._balance_monitor = balance_monitor
        self.__random_state = np.random.RandomState()
        self.__samples_presented = 0

    def initialize_samples(self, sample_locations):
        super().initialize_samples(sample_locations)
        self._generate_sample_content()

    def _balance_monitor_func(self, a):
        if self._balance_monitor == "label":
            return 1
        if self._balance_monitor == "pixel":
            #TODO: THIS WILL OVERWEIGHT LARGE OBJECTS THAT SPAN MORE THAN ONE FRAME
            #TODO: Consider how a sample with both building and roadline should be handled...
            if isinstance(a, Building):
                return a.getGeometry("pixel").area
            if isinstance(a, (LabeledRoadLine, RoadLine)):
                return a.getGeometry("pixel").length
            raise ValueError("Error: Unexpected Spatial Object Found. Expected Either RoadLine or Building.")
        raise ValueError("Error: Balance monitor must be one of label or pixel.")

    def  __len__(self):
        return self._length

    def _spatial_objects_in(self, sample):
        #Flatten a sample into the spatial objects it contains: its buildings plus every
        #labeled sub-line of each of its road lines.
        roadline_sublines = []
        for roadline in sample.getRoadLines():
            for subline in roadline.get_labeled_sub_lines():
                roadline_sublines.append(subline)
        return sample.getBuildings() + roadline_sublines

    def _compute_sample_content(self, sample):
        #The per-class content of a single sample: for each class, the summed balance-monitor
        #value (label count, or pixel area/length) of the sample's objects of that class.
        content = {k: 0.0 for k in self._classes}
        for a in self._spatial_objects_in(sample):
            content[a.getLabel()] += self._balance_monitor_func(a)
        return content

    def _generate_sample_content(self):
        #Precompute a (num_samples, num_classes) matrix of per-class content once, so each draw
        #only needs a single matrix-vector product of this matrix against the class weights.
        self._sample_content = np.zeros((len(self._sample_locations), len(self._classes)))
        for i, sample in enumerate(self._sample_locations):
            content = self._compute_sample_content(sample)
            for j, k in enumerate(self._classes):
                self._sample_content[i, j] = content[k]

    def _add_sample_class_observation_counts(self, sample_location):
        #Accumulate the monitored value (label count or pixel mass) of every spatial object in the
        #presented sample into the running per-class observation counts. We sum per object so that
        #multiple objects sharing a label are all counted (a dict keyed by label would keep only
        #the last one's value, undercounting any class that appears more than once in a sample).
        for a in self._spatial_objects_in(sample_location):
            self._sample_class_observation_counts[a.getLabel()] += self._balance_monitor_func(a)

    def _compute_sample_selection_probabilities(self):
        #First, measure how far below its target proportion each class currently sits. A class at
        #or above target gets zero weight; an under-served class is weighted by its deficit, so the
        #further below target a class is, the harder we pull toward samples that contain it. This is
        #proportional (magnitude-aware): the correction scales with the size of the gap, not just
        #its sign/rank.
        observed_sample_proportions = distribution_proportion(list(self._sample_class_observation_counts.values()))
        observed_sample_proportions_dict = dict(zip(self._sample_class_observation_counts.keys(), observed_sample_proportions))
        class_weights = np.array([
            max(0.0, self._sample_target_class_proportions[k] - observed_sample_proportions_dict[k])
            for k in self._classes
        ])

        #The value of a sample is how much of the currently-deficient classes it carries. Empty
        #samples get zero value and are never drawn. The optional smoothing floor (default 0) adds a
        #small baseline to every non-empty sample for exploration; keep it small, since it biases
        #toward classes that have more sample tiles and so degrades target-matching on imbalanced
        #pools (an all-size-1 pool with many tiles of one class is the worst case).
        non_empty_samples = self._sample_content.sum(axis=1) > 0
        sample_values = self._sample_content @ class_weights + self._sample_selection_smoothing * non_empty_samples

        sample_values_sum = sample_values.sum()
        if sample_values_sum <= 0:
            #No non-empty sample carries a currently-deficient class (e.g. cold start with smoothing=0,
            #or the target is already met): draw uniformly over non-empty samples so empty tiles are
            #never presented. Only if every sample is empty do we fall back to a full uniform draw.
            if non_empty_samples.any():
                return non_empty_samples / non_empty_samples.sum()
            return np.full(len(self._sample_locations), 1.0 / len(self._sample_locations))
        return sample_values / sample_values_sum

    def _get_next_weighted_sample(self):
        sample_selection_probabilities = self._compute_sample_selection_probabilities()
        index = np.random.choice(a=np.arange(0, len(self._sample_locations)), p=sample_selection_probabilities)
        return self._sample_locations[index]

    def getSampleLocation(self, index):
        reseed_distributed(self.__samples_presented, self.__random_state)
        self.__samples_presented += 1

        selected_sample = self._get_next_weighted_sample()
        self._add_sample_class_observation_counts(selected_sample)
        return selected_sample

#Below are the sample location presentation strategies to be used when the sample locations are not known ahead of time
class RealTimeSampleLocationPresentationStrategy(SampleLocationPresentationStrategy):
    def observeSampleLocation(self, sample_location):
        raise NotImplementedError("Function observeSample must be implemented by a subclass.")
    def getSampleLocation(self, index):
        raise NotImplementedError("Function getSampleLocation must be implemented by a subclass.")

class MostRecentlyObservedSampleLocationPresentationStrategy(RealTimeSampleLocationPresentationStrategy):
    def __init__(self, length):
        super().__init__()
        self._length = length
        self._most_recent_sample = None
    def getSampleLocation(self, index):
        return self._most_recent_sample
    def observeSampleLocation(self, sample_location):
        self._most_recent_sample = sample_location
    def __len__(self):
        return self._length
