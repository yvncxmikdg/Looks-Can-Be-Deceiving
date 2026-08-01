from modeling.Models.TaskModuleBDA import TaskModuleBDA


class TaskModuleCHANGE(TaskModuleBDA):
    """Cross-source building-change task: BDA's per-building machinery over CHANGED/UNCHANGED.

    Training, validation, prediction, and fusion are inherited from TaskModuleBDA unchanged; the
    only difference is the label set the per-building fusion runs over, which comes from the CHANGE
    channel map instead of the BDA damage classes.
    """

    def _fusion_class_labels(self):
        background_labels = set(self.output_label_map.getBackgroundClass())
        return [label for label in self.output_label_map.getAllLabels() if label not in background_labels]
