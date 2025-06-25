from transformers import (Trainer, TrainingArguments, AutoModelForCausalLM,
                          DataCollatorForLanguageModeling, AutoTokenizer, GPT2Config,
                          TrainerCallback)
from torch.utils.data import DataLoader
import torch.nn.functional as F

import math
import os
import shutil
import sys
import time
from typing import Any, Dict, Union, Optional, List, Tuple

from tqdm.auto import tqdm


# Integrations must be imported before ML frameworks:
from transformers.integrations import (  # isort: split
    hp_params,
)
import torch
import torch.distributed as dist
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.modeling_utils import unwrap_model
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.deepspeed import deepspeed_init
from transformers.pytorch_utils import is_torch_greater_or_equal_than_1_10, is_torch_less_than_1_11
from transformers.trainer_callback import (
    DefaultFlowCallback,
    ProgressCallback,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    IterableDatasetShard,
    nested_detach
)
from transformers.trainer_utils import (
    HPSearchBackend,
    ShardedDDPOption,
    TrainOutput,
    has_length,
    seed_worker,
    speed_metrics,
)
from transformers.utils import (
    is_apex_available,
    is_in_notebook,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
    logging,
)

_is_native_cpu_amp_available = is_torch_greater_or_equal_than_1_10

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
    from apex import amp

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

logger = logging.get_logger(__name__)

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"


class TrainerWithSyntacticRegularizer(Trainer):
    def __init__(
            self,
            model = None,
            args = None,
            data_collator = None,
            train_dataset = None,
            eval_dataset = None,
            tokenizer = None,
            model_init = None,
            compute_metrics = None,
            callbacks = None,
            optimizers = (None, None),
            preprocess_logits_for_metrics = None,
            reg_lambda = 0.001,
            device=None
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
        )
        self.reg_lambda = reg_lambda
        self.device = device

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self._train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )

            return DataLoader(
                train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        train_sampler = self._get_train_sampler()

        return DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=seed_worker,
        )

    def _to_word_level_attentions(self, attentions, tok2word, pad=True):
        """

        :param attentions: batch of attentions; [torch(shape: (layer, head, source, target)), ...]
        :return:
        """
        # original: (layer, batch_size, head, source, target)
        # reshaped: (batch_size, layer, head, source, target)
        attentions = torch.stack(attentions, dim=1)
        word_level_attentions = []
        ctx_size = attentions.shape[-1]
        for seq_id, seq in enumerate(attentions):
            # seq has size: (layer, head, source token, target token)

            ### all "attended to" tokens should be *summed* over constituent tokens to map to word level
            # word id each tok corresponds to e.g. [0, 0, 1, 1, 1, 2, 3, ...] (first 2 toks correspond to word 0)
            word_membership = tok2word[seq_id]
            # number of words (<=ctx_size)
            # num_words = len(set(word_membership.tolist()))
            num_words = word_membership[-1]+1
            # reshape the membership tensor to match the source tensor (original token level attention tensor)
            word_membership = word_membership.expand(*seq.shape[:-1], -1)
            # initialize the word level tensor to all zeros
            target_summed = torch.zeros(*seq.shape[:-1], num_words, device=self.device)
            # torch.scatter_add(input=test, dim=-1, index=assignment, src=summed)
            target_summed.scatter_add_(dim=-1, index=word_membership, src=seq)  # (layer, head, source_token, target_word)

            ### all "attended from" tokens should be *averaged* over constituent tokens to map to word level
            word_membership = tok2word[seq_id]
            target_summed = target_summed.permute(0, 1, 3, 2)  # (layer, head, target=word, source=tok)
            word_membership = word_membership.expand(*target_summed.shape[:-1], -1)  # layer, head, word, tok

            # initialize the word level tensor to all zeros
            source_averaged = torch.zeros(*target_summed.shape[:-2], num_words, num_words, device=self.device)
            # torch.scatter_add(input=test, dim=-1, index=assignment, src=summed)
            source_averaged.scatter_add_(dim=-1, index=word_membership,
                                         src=target_summed)  # (layer, head, source_token, target_word)
            word_membership_counts = torch.zeros_like(source_averaged, device=self.device)  # initialize constituent token count vector
            word_membership_counts.scatter_add_(
                dim=-1,
                index=word_membership,
                src=torch.ones_like(target_summed)
            )
            source_averaged = source_averaged / word_membership_counts
            source_averaged = source_averaged.permute(0, 1, 3, 2)  # back to the original
            # source_averaged.to('cpu')  # offload it to CPU
            if pad:
                pad_amount = ctx_size - source_averaged.shape[-1]
                source_averaged = F.pad(
                    source_averaged,
                    pad=(0, pad_amount, 0, pad_amount),
                    mode='constant',
                    value=0
                )
            word_level_attentions.append(source_averaged)

        return word_level_attentions

    # def _get_reg_val(self, word_level_attentions, heads):
    #     # Stack word_level_attentions along a new dimension
    #     attentions = torch.stack(word_level_attentions, dim=0)  # (batch_size, layers, num_tok, num_tok)
    #
    #     # Step 1: Get max along layers and sequence dimensions
    #     max_seq = torch.max(attentions, dim=2).values  # (batch_size, layers, num_tok)
    #     seq = torch.max(max_seq, dim=1).values  # (batch_size, num_tok)
    #
    #     # Step 2: Prepare heads tensor and filter
    #     num_words = seq.shape[-1]
    #     heads = torch.where(heads < num_words, heads, -1)
    #
    #     # Step 3: Create pairs for all sequences
    #     indices = torch.arange(heads.size(1), device=heads.device).expand_as(heads)
    #     pairs = torch.where(
    #         indices.unsqueeze(2) >= heads.unsqueeze(2),
    #         torch.stack([indices, heads], dim=2),  # (batch_size, num_heads, 2)
    #         torch.stack([heads, indices], dim=2)
    #     )
    #
    #     # Step 4: Filter pairs containing -1
    #     valid_mask = ~torch.any(pairs == -1, dim=-1)  # (batch_size, num_words)
    #     valid_pairs = pairs[valid_mask].reshape(-1, 2)  # Collect valid pairs
    #
    #     # Step 5: Create indices tensor for loss calculation
    #     indices = torch.zeros_like(seq)
    #     indices[:, valid_pairs[:, 0], valid_pairs[:, 1]] = 1
    #
    #     # Step 6: Calculate loss for all sequences
    #     reg = torch.sum(seq * indices)
    #
    #     return reg

    def _get_reg_val(self, word_level_attentions, heads):

        # Stack word_level_attentions along a new dimension
        attentions = torch.stack(word_level_attentions, dim=0)  # (batch_size, layers, num_tok, num_tok)

        # Step 1: Get max attention scores along layers and sequence dimensions
        seq = attentions.amax(dim=(1, 2))  # (batch_size, num_tok)
        batch_size, ctx_size, _ = seq.shape

        # set valid_mask to be 0; -1 is always the target (source >= target), so setting those 0 should suffice
        valid_mask = (heads >= 0) & (heads < ctx_size)  # Mask invalid head positions
        valid_mask = valid_mask.unsqueeze(1).expand(batch_size, ctx_size, ctx_size)
        heads = torch.clamp(heads, min=0, max=ctx_size-1)

        # Step 2: Create index pairs for valid heads
        indices = torch.arange(ctx_size, device=heads.device).unsqueeze(0)  # (1, num_words)

        # Stack all possible (i, j) pairs
        pairs = torch.where(
            indices.unsqueeze(2) >= heads.unsqueeze(2),
            torch.stack([indices.expand(batch_size, ctx_size), heads], dim=-1),
            torch.stack([heads, indices.expand(batch_size, ctx_size)], dim=-1)
        )

        # Step 3: Create the indices tensor for loss calculation
        indices_tensor = torch.zeros_like(seq)  # Initialize a zero tensor (batch_size, num_tok)

        # Set the value to 1
        batch_indices = torch.arange(indices_tensor.shape[0]).unsqueeze(1)
        row = pairs[..., 0]
        col = pairs[..., 1]
        indices_tensor[batch_indices, row, col] = 1

        # Step 5: Apply both masks (dependency edge mask, valid mask)
        reg = torch.sum(seq * indices_tensor * valid_mask)

        return reg

    def compute_loss(self, model, inputs, return_outputs=False, regularize=True):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        # print(inputs["input_ids"][:10])
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
            output_attentions=True,
            return_dict=True,
        )
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        lm_loss, reg = None, None
        if regularize:
            lm_loss = outputs["loss"].clone() if isinstance(outputs, dict) else outputs[0]
            # reg = outputs["loss"].clone() if isinstance(outputs, dict) else outputs[0]
            # Now add dependency parsing regularization - suppresses attention weights corresponding to the head
            # of each dependency relation
            attentions = self._to_word_level_attentions(outputs.attentions, inputs['tok2word'])
            heads = inputs["heads"]
            # Get rows in batch that have "heads" and "relns" populated
            reg = self._get_reg_val(attentions, heads)
            assert reg.requires_grad
            loss += self.reg_lambda * reg

        if return_outputs:
            return loss, outputs, lm_loss, reg
        else:
            return loss, lm_loss, reg
        # return (loss, outputs, lm_loss, reg) if return_outputs else loss, lm_loss, reg

    def _compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
    ):
        """
        Expects inputs to include "heads" key too.
        """
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
            # input_ids=inputs["input_ids"].cuda(),
            # attention_mask=inputs["attention_mask"].cuda(),
            # labels=inputs["labels"].cuda(),
            output_attentions=True,
            return_dict=True,
        )
        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        lm_loss = outputs["loss"].clone() if isinstance(outputs, dict) else outputs[0]
        # Now add dependency parsing regularization - suppresses attention weights corresponding to the head
        # of each dependency relation
        attentions = self._to_word_level_attentions(outputs.attentions, inputs['tok2word'])
        heads = inputs["heads"]
        # Get rows in batch that have "heads" and "relns" populated
        reg = self._get_reg_val(attentions, heads)
        assert reg.requires_grad
        loss += self.reg_lambda * reg

        return (loss, outputs, lm_loss, reg) if return_outputs else loss, lm_loss, reg


    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self._train_batch_size = batch_size
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = (
            self.sharded_ddp is not None
            and self.sharded_ddp != ShardedDDPOption.SIMPLE
            or is_sagemaker_mp_enabled()
            or self.fsdp is not None
        )
        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")
        logger.info(
            f"  Number of trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        tr_lm_loss = torch.tensor(0.0).to(args.device)
        tr_reg = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                is_random_sampler = hasattr(train_dataloader, "sampler") and isinstance(
                    train_dataloader.sampler, RandomSampler
                )
                if is_torch_less_than_1_11 or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    # That was before PyTorch 1.11 however...
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    _ = list(train_dataloader.sampler)

        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            step = -1
            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                if (
                    ((step + 1) % args.gradient_accumulation_steps != 0)
                    and args.local_rank != -1
                    and args._no_sync_in_gradient_accumulation
                ):
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    with model.no_sync():
                        tr_loss_step, tr_lm_loss_step, tr_reg_step = self.training_step(model, inputs)
                else:
                    tr_loss_step, tr_lm_loss_step, tr_reg_step = self.training_step(model, inputs)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                    tr_lm_loss += tr_lm_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                    tr_reg += tr_reg / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step
                    tr_lm_loss += tr_lm_loss_step
                    tr_reg += tr_reg_step

                self.current_flos += float(self.floating_point_ops(inputs))

                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if self.deepspeed:
                    self.deepspeed.step()

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                        # deepspeed does its own clipping

                        if self.do_grad_scaling:
                            # Reduce gradients first for XLA
                            if is_torch_tpu_available():
                                gradients = xm._fetch_gradients(self.optimizer)
                                xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)

                        if is_sagemaker_mp_enabled() and args.fp16:
                            self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    optimizer_was_run = True
                    if self.deepspeed:
                        pass  # called outside the loop
                    elif is_torch_tpu_available():
                        if self.do_grad_scaling:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            xm.optimizer_step(self.optimizer)
                    elif self.do_grad_scaling:
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        optimizer_was_run = scale_before <= scale_after
                    else:
                        self.optimizer.step()

                    if optimizer_was_run and not self.deepspeed:
                        self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, tr_lm_loss, tr_reg, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, tr_lm_loss, tr_reg, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.local_rank != -1:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint.
        if self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if checkpoint != self.state.best_model_checkpoint:
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def training_step(self,
                      model: nn.Module,
                      inputs: Dict[str, Union[torch.Tensor, Any]]
                      ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss, lm_loss, reg = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
            lm_loss = lm_loss.mean()
            reg = reg.mean()

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps
            lm_loss = lm_loss / self.args.gradient_accumulation_steps
            reg = reg / self.args.gradient_accumulation_steps
        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach(), lm_loss.detach(), reg.detach()

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels or loss_without_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels or loss_without_labels:
                    with self.compute_loss_context_manager():
                        loss, outputs, _, _ = self.compute_loss(model, inputs, return_outputs=True,
                                                                        regularize=False)
                    loss = loss.mean().detach()

                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    with self.compute_loss_context_manager():
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    else:
                        logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)

    def _maybe_log_save_evaluate(self, tr_loss, tr_lm_loss, tr_reg, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            if is_torch_tpu_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
            tr_lm_loss_scalar = self._nested_gather(tr_lm_loss).mean().item()
            tr_reg_scalar = self._nested_gather(tr_reg).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss
            tr_lm_loss -= tr_lm_loss
            tr_reg -= tr_reg

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["lm_loss"] = round(tr_lm_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["syn_reg"] = round(tr_reg_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            if isinstance(self.eval_dataset, dict):
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                    )
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)


    # def create_scheduler(self, num_training_steps: int, optimizer: Optimizer = None):
    #     """
    #     Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
    #     passed as an argument.
    #
    #     >> IMPORTANT: Overrides the HF LR scheduler so that num_training_steps is always set to 1M, which works
    #     >> better for multi-stage training jobs.
    #
    #     Args:
    #         num_training_steps (int): The number of training steps to do.
    #         optimizer (torch.optim.Optimizer): Optimizer to wrap in the scheduler.
    #     """
    #     if self.lr_scheduler is None:
    #         opt = self.optimizer if optimizer is None else optimizer
    #         self.lr_scheduler = self._get_linear_schedule_with_warmup(
    #             opt,
    #             self.args.get_warmup_steps(num_training_steps),
    #         )
    #     return self.lr_scheduler
    #
    # def _get_linear_schedule_with_warmup(
    #     self, optimizer, num_warmup_steps, last_epoch=-1
    # ):
    #     """
    #     Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    #     a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    #
    #     Args:
    #         optimizer (:class:`~torch.optim.Optimizer`):
    #             The optimizer for which to schedule the learning rate.
    #         num_warmup_steps (:obj:`int`):
    #             The number of steps for the warmup phase.
    #         num_training_steps (:obj:`int`):
    #             The total number of training steps.
    #         last_epoch (:obj:`int`, `optional`, defaults to -1):
    #             The index of the last epoch when resuming training.
    #
    #     Return:
    #         :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    #     """
    #     num_training_steps = 10**6
    #
    #     def lr_lambda(current_step: int):
    #         if current_step < num_warmup_steps:
    #             return float(current_step) / float(max(1, num_warmup_steps))
    #         return max(
    #             0.0,
    #             float(num_training_steps - current_step)
    #             / float(max(1, num_training_steps - num_warmup_steps)),
    #         )
    #
    #     return LambdaLR(optimizer, lr_lambda, last_epoch)

"""TEST"""

# from datasets import load_dataset
# class CustomSaveCallback(TrainerCallback):
#     def __init__(self, steps_to_save):
#         self.steps_to_save = set(steps_to_save)
#
#     def on_step_end(self, args, state, control, **kwargs):
#         print(f"Current step: {state.global_step}")
#         if state.global_step in self.steps_to_save:
#             # Save the model
#             output_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
#             os.makedirs(output_dir, exist_ok=True)
#             kwargs['model'].save_pretrained(output_dir)
#             kwargs['tokenizer'].save_pretrained(output_dir)
#             print(f"Model saved at step {state.global_step}")
#
#
# model = AutoModelForCausalLM.from_pretrained('gpt2')
# tokenizer = AutoTokenizer.from_pretrained('gpt2')
# config = GPT2Config(n_ctx=1024,
#                     n_positions=1024,
#                     eos_token_id=tokenizer.eos_token_id,
#                     n_embd=16,
#                     n_head=4,
#                     n_layer=2
#                     )
# model = AutoModelForCausalLM.from_config(config)
# model.to('cpu')
# tokenizer.pad_token = tokenizer.eos_token
# data_fp = os.path.join('data', 'pile_1b_tokens_heads.jsonl')
# data = load_dataset('json', data_files=data_fp, streaming=True, split='train')
# data = data.with_format('torch')
#
# training_args = TrainingArguments(
#     report_to=None,
#     output_dir=os.path.join(os.getcwd(), 'models', 'testt'),
#     overwrite_output_dir=False,
#     do_train=True,
#     do_eval=False,
#     do_predict=False,
#     max_steps=4,
#     save_steps=10,  # avoiding saving besides the custom saving
#     logging_steps=10,
#     learning_rate=2.5e-4,
#     lr_scheduler_type='cosine',
#     warmup_steps=0,
#     per_device_train_batch_size=1,
#     gradient_accumulation_steps=2,
#     no_cuda=True
# )
# trainer = TrainerWithSyntacticRegularizer(
#     model=model,
#     tokenizer=tokenizer,
#     args=training_args,
#     data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
#     train_dataset=data,
#     callbacks=[CustomSaveCallback([1,2])]
# )
# trainer.train()