# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Defining how components interface with each other."""
from argparse import Namespace
from typing import Type

import launchpad as lp
from launchpad.nodes.python import local_multi_processing

from oat.actor import Actor
from oat.learners.base import LearnerBase
from oat.utils.ipc import PlasmaShmServer
from oat.utils.launcher import get_free_port


def get_program(
    args: Namespace, learner_cls: Type[LearnerBase], actor_cls: Type[Actor] = Actor
):
    """Define the default distributed program topology with configs."""
    program = lp.Program("online_dap")

    # Resource.
    if args.total_gpus == 2:
        actor_gpus = [0]
        learner_gpus = [1]
    elif args.total_gpus == 3:
        actor_gpus = [0, 1]
        learner_gpus = [2, 1]
    elif args.total_gpus == 4:
        actor_gpus = [0, 1]
        learner_gpus = [2, 3]
        if args.collocate:
            actor_gpus = [0, 1, 2]
            learner_gpus = [3, 2, 1]
    elif args.total_gpus == 5:
        actor_gpus = [0, 1, 2, 3]
        learner_gpus = [4, 3, 2, 1]
    elif args.total_gpus == 6:
        actor_gpus = [0, 1, 2]
        learner_gpus = [3, 4, 5]
        if args.collocate:
            actor_gpus = [0, 1, 2, 3]
            learner_gpus = [4, 3, 2, 1]
    elif args.total_gpus == 8:
        actor_gpus = [0, 1, 2, 3]
        learner_gpus = [4, 5, 6, 7]
        if args.collocate:
            actor_gpus = [0, 1, 2, 3, 4, 5, 6]
            learner_gpus = [7, 6, 5, 4, 3, 2, 1]
    else:
        assert False

    # IPC.
    ipc_server = program.add_node(
        lp.CourierNode(PlasmaShmServer, size_mb=args.shm_size_mb), label="ipc_server"
    )

    # Actor.
    vllm_args = {
        "model": args.pretrain,
        "trust_remote_code": True,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": args.vllm_gpu_ratio,
        "dtype": "bfloat16",
        "enable_prefix_caching": False,
    }

    actors = []
    local_resources = {}
    for i in actor_gpus:
        label = f"actor_{i}"
        actors.append(
            program.add_node(
                lp.CourierNode(actor_cls, ipc_server, vllm_args, args),
                label=label,
            )
        )
        local_resources[label] = local_multi_processing.PythonProcess(
            env={"CUDA_VISIBLE_DEVICES": str(i)}
        )

    # Learner.
    master_addr = "0.0.0.0"
    master_port = get_free_port()
    args.local_rank = 0
    label = "learner_0"
    master_learner = lp.PyClassNode(
        learner_cls,
        len(learner_gpus),
        0,
        0,
        master_addr,
        master_port,
        True,
        args,
        actors,
        ipc_server,
    )
    program.add_node(master_learner, label=label)
    local_resources[label] = local_multi_processing.PythonProcess(
        env={"CUDA_VISIBLE_DEVICES": str(learner_gpus[0])}
    )
    for i in range(1, len(learner_gpus)):
        label = f"learner_{i}"
        worker_learner = lp.PyClassNode(
            learner_cls,
            len(learner_gpus),
            i,
            i,
            master_addr,
            master_port,
            False,
            args,
            actors,
            ipc_server,
        )
        program.add_node(worker_learner, label=label)
        local_resources[label] = local_multi_processing.PythonProcess(
            env={"CUDA_VISIBLE_DEVICES": str(learner_gpus[i])}
        )

    return program, local_resources
