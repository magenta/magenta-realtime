# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Top-level pytest configuration for the test suite.

pytest imports this module before collecting any test module, so we use it to
activate the vendor hook (`magenta_rt._vendor._vendor_hook`) up front. The hook
puts the bundled `audiotree` and `sequence_layers` packages on `sys.path` when
they are not pip-installed. Several test modules do `from audiotree import
AudioTree` *before* importing `magenta_rt` (which would otherwise run the hook),
so without this they only collect when some other module happened to import
`magenta_rt` first — running such a file alone fails with `ModuleNotFoundError`.
Installing the hook here makes the vendored imports work regardless of test
order or which file is selected.
"""

from magenta_rt._vendor import _vendor_hook

_vendor_hook.install()
