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

"""Offline alignment with online vLLM evaluation."""

import launchpad as lp
from ellm.args import default_args_validation, get_default_parser
from ellm.interface import get_program
from ellm.learners import OfflineDAPLearner


def main(args):
    program, local_resources = get_program(args, OfflineDAPLearner)
    lp.launch(
        program,
        launch_type="local_mp",
        local_resources=local_resources,
        terminal="current_terminal",
    )


if __name__ == "__main__":
    parser = get_default_parser()
    parser.add_argument("--preference_data", type=str, default="")
    parser.add_argument("--prompt_key", type=str, default="prompt")
    parser.add_argument("--chosen_key", type=str, default="chosen")
    parser.add_argument("--rejected_key", type=str, default="rejected")
    parser.add_argument("--offline_buffer_path", type=str, default="./data/buffer.pkl")
    args = default_args_validation(parser.parse_args())
    main(args)
