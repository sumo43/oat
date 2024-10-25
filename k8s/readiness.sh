#!/bin/bash
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


URL=http://localhost:8000/compare 
STATUS_CODE=$(curl -X POST -H "Content-Type: application/octet-stream" --data-binary $'\x82\xa6prompt\xa2Hi\xaacandidates\x92\xa6Hello!\xa4Hey!' -o /dev/null -s -w "%{http_code}" "$URL")

if [ "$STATUS_CODE" -eq 200 ]; then
  echo "Success"
  exit 0  # success
else
  echo "Failure"
  exit 1  # failure
fi