import logging
from typing import TYPE_CHECKING, List, Tuple, Union

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS
from sglang.srt.managers.io_struct import (
    GetWeightsByNameReqInput,
    GetWeightsByNameReqOutput,
    InitWeightsUpdateGroupReqInput,
    InitWeightsUpdateGroupReqOutput,
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
    ResumeMemoryOccupationReqInput,
    ResumeMemoryOccupationReqOutput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightFromDiskReqOutput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromDistributedReqOutput,
    UpdateWeightsFromTensorReqInput,
    UpdateWeightsFromTensorReqOutput,
)

logger = logging.getLogger(__name__)


class SchedulerUpdateWeightsMixin:

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        """In-place update of the weights from disk."""
        success, message = self.tp_worker.update_weights_from_disk(recv_req)
        if success:
            flush_cache_success = self.flush_cache()
            assert flush_cache_success, "Cache flush failed after updating weights"
        else:
            logger.error(message)
        return UpdateWeightFromDiskReqOutput(success, message, 0)

    def init_weights_update_group(self, recv_req: InitWeightsUpdateGroupReqInput):
        """Initialize the online model parameter update group."""
        success, message = self.tp_worker.init_weights_update_group(recv_req)
        return InitWeightsUpdateGroupReqOutput(success, message)

    def update_weights_from_distributed(
        self,
        recv_req: UpdateWeightsFromDistributedReqInput,
    ) -> Tuple[bool, str]:
        """Update the online model parameter."""
        success, message = self.tp_worker.update_weights_from_distributed(recv_req)
        if success:
            if recv_req.flush_cache:
                flush_cache_success = self.flush_cache()
                assert flush_cache_success, "Cache flush failed after updating weights"
        else:
            logger.error(message)
        return UpdateWeightsFromDistributedReqOutput(success, message)

    def _split_worker_and_draft_weight(
        self,
        named_tensors: Union[
            "FlattenedTensorBucketDict",
            List[Tuple[str, Union[torch.Tensor, "LocalSerializedTensor"]]],
        ],
    ):
        if isinstance(named_tensors, list):
            # List of tuples: (name, tensor)
            tp_worker_named_tensors = [
                (name, tensor)
                for name, tensor in named_tensors
                if name in self.tp_worker_param_names
            ]
            if self.draft_worker is not None:
                draft_worker_named_tensors = [
                    (name, tensor)
                    for name, tensor in named_tensors
                    if name not in self.tp_worker_param_names
                ]
            else:
                draft_worker_named_tensors = None
            return tp_worker_named_tensors, draft_worker_named_tensors
        elif isinstance(named_tensors, dict):
            flattend_tensor = named_tensors["flattened_tensor"]
            total_meta = named_tensors["metadata"]
            tp_meta = [
                meta for meta in total_meta if meta.name in self.tp_worker_param_names
            ]
            tp_worker_named_tensors = {
                "flattened_tensor": flattend_tensor,
                "metadata": tp_meta,
            }
            if self.draft_worker is not None:
                draft_meta = [
                    meta
                    for meta in total_meta
                    if meta.name not in self.tp_worker_param_names
                ]
                draft_worker_named_tensors = {
                    "flattened_tensor": flattend_tensor,
                    "metadata": draft_meta,
                }
            else:
                draft_worker_named_tensors = None
            return tp_worker_named_tensors, draft_worker_named_tensors
        else:
            raise ValueError(f"Invalid type for named_tensors: {type(named_tensors)}")

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        """Update the online model parameter from tensors."""
        success, message = self.tp_worker.update_weights_from_tensor(recv_req)
        if self.draft_worker is not None and hasattr(
            self.draft_worker, "update_weights_from_tensor"
        ):
            draft_success, draft_message = self.draft_worker.update_weights_from_tensor(
                recv_req
            )
            success = success and draft_success
            message = f"Main model: {message}. Draft model: {draft_message}."
        # TODO extract common code b/t update_weights_from_distributed and update_weights_from_tensor later
        if success:
            if recv_req.flush_cache:
                flush_cache_success = self.flush_cache()
                assert flush_cache_success, "Cache flush failed after updating weights"
        else:
            logger.error(message)
        torch.distributed.barrier(group=self.tp_cpu_group)
        return UpdateWeightsFromTensorReqOutput(success, message)

    def get_weights_by_name(self, recv_req: GetWeightsByNameReqInput):
        parameter = self.tp_worker.get_weights_by_name(recv_req)
        return GetWeightsByNameReqOutput(parameter)

    def release_memory_occupation(self, recv_req: ReleaseMemoryOccupationReqInput):
        tags = recv_req.tags

        if tags is None or len(tags) == 0:
            tags = [GPU_MEMORY_TYPE_WEIGHTS, GPU_MEMORY_TYPE_KV_CACHE]

        for tag in tags:
            self.offload_tags.add(tag)

        if GPU_MEMORY_TYPE_KV_CACHE in tags:
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_KV_CACHE)
            self.flush_cache()

        if GPU_MEMORY_TYPE_WEIGHTS in tags:
            self.stashed_model_static_state = _export_static_state(
                self.tp_worker.worker.model_runner.model
            )
            torch.distributed.barrier(self.tp_cpu_group)
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_WEIGHTS)

        return ReleaseMemoryOccupationReqOutput()

    def resume_memory_occupation(self, recv_req: ResumeMemoryOccupationReqInput):
        tags = recv_req.tags

        if tags is None or len(tags) == 0:
            tags = [GPU_MEMORY_TYPE_WEIGHTS, GPU_MEMORY_TYPE_KV_CACHE]

        for tag in tags:
            self.offload_tags.remove(tag)

        if GPU_MEMORY_TYPE_WEIGHTS in tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_WEIGHTS)
            torch.distributed.barrier(self.tp_cpu_group)
            _import_static_state(
                self.tp_worker.worker.model_runner.model,
                self.stashed_model_static_state,
            )
            del self.stashed_model_static_state

        if GPU_MEMORY_TYPE_KV_CACHE in tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_KV_CACHE)

        return ResumeMemoryOccupationReqOutput()

    def save_remote_model(self, params):
        url = params["url"]

        worker = self.tp_worker.worker
        worker.model_runner.save_remote_model(url)

        if self.draft_worker is not None:
            draft_url = params.get("draft_url", None)
            assert (
                draft_url is not None
            ), "draft_url must be provided when draft model is enabled"
            draft_worker = self.draft_worker.worker
            draft_worker.model_runner.save_remote_model(draft_url)

    def save_sharded_model(self, params):
        worker = self.tp_worker.worker

        worker.model_runner.save_sharded_model(
            path=params["path"],
            pattern=params["pattern"],
            max_size=params["max_size"],
        )


def _export_static_state(model):
    return dict(
        buffers=[
            (name, buffer.detach().clone()) for name, buffer in model.named_buffers()
        ]
    )


def _import_static_state(model, static_params):
    self_named_buffers = dict(model.named_buffers())
    for name, tensor in static_params["buffers"]:
        self_named_buffers[name][...] = tensor
