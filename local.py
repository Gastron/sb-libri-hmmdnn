import torch
import speechbrain.utils.checkpoints

class NoSchedule:
    def __init__(self, lr, **kwargs):
        self.lr = lr
        self.current_lr = lr

    def __call__(self, *args, **kwargs):
        return self.lr, self.lr

@checkpoints.register_checkpoint_hooks
class NewBobSchedulerWithWarmup:
    """Scheduler with new-bob technique, used for LR annealing.
    The learning rate is annealed based on the validation performance.
    In particular: if (past_loss-current_loss)/past_loss< impr_threshold:
    lr=lr * annealing_factor.
    Arguments
    ---------
    initial_value : float
        The initial hyperparameter value.
    annealing_factor : float
        It is annealing factor used in new_bob strategy.
    improvement_threshold : float
        It is the improvement rate between losses used to perform learning
        annealing in new_bob strategy.
    patient : int
        When the annealing condition is violated patient times,
        the learning rate is finally reduced.
    Example
    -------
    >>> scheduler = NewBobScheduler(initial_value=1.0)
    >>> scheduler(metric_value=10.0)
    (1.0, 1.0)
    >>> scheduler(metric_value=2.0)
    (1.0, 1.0)
    >>> scheduler(metric_value=2.5)
    (1.0, 0.5)
    """

    def __init__(
        self,
        initial_value,
        highest_value,
        warmup_epochs=10,
        annealing_factor=0.5,
        improvement_threshold=0.0025,
        patient=0,
    ):
        self.hyperparam_value = initial_value
        self.annealing_factor = annealing_factor
        self.improvement_threshold = improvement_threshold
        self.patient = patient
        self.metric_values = []
        self.current_patient = self.patient
        self.curr_epoch = 1
        self.initial_value = initial_value
        self.highest_value = highest_value
        self.warmup_epochs = warmup_epochs

    def __call__(self, metric_value):
        """Returns the current and new value for the hyperparameter.
        Arguments
        ---------
        metric_value : int
            A number for determining whether to change the hyperparameter value.
        """
        old_value = new_value = self.hyperparam_value
        next_epoch = self.curr_epoch + 1
        if self.next_epoch <= self.warmup_epochs:
            coeff = (next_epoch - 1)/(self.warmup_epochs - 1)
            new_value = coeff * self.highest_value + (1 - coeff)
        elif len(self.metric_values) > 0:
            prev_metric = self.metric_values[-1]
            # Update value if improvement too small and patience is 0
            if prev_metric == 0:  # Prevent division by zero
                improvement = 0
            else:
                improvement = (prev_metric - metric_value) / prev_metric
            if improvement < self.improvement_threshold:
                if self.current_patient == 0:
                    new_value *= self.annealing_factor
                    self.current_patient = self.patient
                else:
                    self.current_patient -= 1

        # Store relevant info
        self.metric_values.append(metric_value)
        self.hyperparam_value = new_value

        return old_value, new_value

    @checkpoints.mark_as_saver
    def save(self, path):
        """Saves the current metrics on the specified path."""
        data = {
            "hyperparam_value": self.hyperparam_value,
            "metric_values": self.metric_values,
            "current_patient": self.current_patient,
            "curr_epoch": self.curr_epoch,
        }
        torch.save(data, path)

    @checkpoints.mark_as_loader
    def load(self, path, end_of_epoch=False, device=None):
        """Loads the needed information."""
        del end_of_epoch  # Unused in this class
        del device  # Unused in here
        data = torch.load(path)
        self.hyperparam_value = data["hyperparam_value"]
        self.metric_values = data["metric_values"]
        self.current_patient = data["current_patient"]
        self.curr_epoch = data["curr_epoch"]
