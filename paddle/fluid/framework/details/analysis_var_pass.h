// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <set>
#include <string>

#include "paddle/fluid/framework/details/cfg_graph.h"
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace details {

class AnalysisVarPass : public ir::Pass {
 protected:
  std::unique_ptr<ir::Graph> ApplyImpl(
      std::unique_ptr<ir::Graph> graph) const override;

 private:
  // search pool for a best fit Node.
  bool NodeMatching(ir::Node* var, ir::Node** cache, int* idx) const;

  const std::string DebugString(ir::Node* var) const;
  // Reuse Node Pool
  mutable details::UnlivedNodePool pool;
};

}  // namespace details
}  // namespace framework
}  // namespace paddle