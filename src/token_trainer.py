"""Token-normalized Trainer for the single-device experiment."""

from transformers import Trainer

from src.training_objective import causal_nll_sum, shifted_target_count


class TokenLossTrainer(Trainer):
    """Use accumulated target counts and log token-weighted NLL."""

    def __init__(self, **kwargs):
        super().__init__(compute_loss_func=self._token_loss, **kwargs)
        if self.args.world_size != 1 or self.args.n_gpu > 1:
            raise ValueError("TokenLossTrainer supports exactly one device.")
        if self.accelerator.gradient_accumulation_steps != 1:
            raise ValueError("Trainer must own accumulation. Accelerator must not rescale loss.")
        # The callback owns loss kwargs. Pass none of them to the PEFT forward.
        self.model_accepts_loss_kwargs = False
        self._train_totals = None
        self._window_totals = None
        self._eval_totals = None

    def _get_num_items_in_batch(self, batch_samples, device):
        if batch_samples:
            return sum(shifted_target_count(batch["labels"]) for batch in batch_samples).to(device)
        return None

    @staticmethod
    def _add(totals, loss, count):
        stats = (loss.detach().double(), count.detach())
        if totals is None:
            return stats
        return totals[0] + stats[0], totals[1] + stats[1]

    @staticmethod
    def _mean(totals):
        return (totals[0] / totals[1]).item()

    def _token_loss(self, outputs, labels, num_items_in_batch=None):
        count = shifted_target_count(labels)
        nll = causal_nll_sum(outputs.logits, labels)
        if self.model.training:
            if num_items_in_batch is None:
                raise RuntimeError("Trainer did not supply an accumulated target count.")
            self._train_totals = self._add(self._train_totals, nll, count)
            self._window_totals = self._add(self._window_totals, nll, count)
            return nll / num_items_in_batch
        self._eval_totals = self._add(self._eval_totals, nll, count)
        return nll / count

    def evaluate(self, *args, **kwargs):
        self._eval_totals = None
        return super().evaluate(*args, **kwargs)

    def log(self, logs, start_time=None):
        if "loss" in logs and self._window_totals is not None:
            logs["loss"] = self._mean(self._window_totals)
            self._window_totals = None
        if "train_loss" in logs and self._train_totals is not None:
            logs["train_loss"] = self._mean(self._train_totals)
        for name in tuple(logs):
            if name.endswith("_loss") and name != "train_loss" and self._eval_totals is not None:
                logs[name] = self._mean(self._eval_totals)
        return super().log(logs, start_time=start_time)

    @property
    def training_token_nll(self):
        return self._mean(self._train_totals)
