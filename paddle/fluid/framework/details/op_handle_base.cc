//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/details/op_handle_base.h"

namespace paddle {
namespace framework {
namespace details {
std::string OpHandleBase::DebugString() const {
  std::stringstream ss;
  ss << "(";
  for (auto *var : inputs_) {
    ss << var->DebugString() << ", ";
  }
  ss << ") --> (";
  for (auto *var : outputs_) {
    ss << var->DebugString() << ", ";
  }
  ss << ")\n";
  return ss.str();
}

OpHandleBase::~OpHandleBase() {
#ifdef PADDLE_WITH_CUDA
  for (auto &ev : events_) {
    PADDLE_ENFORCE(cudaEventDestroy(ev.second));
  }
#elif defined(PADDLE_WITH_HIP)
  for (auto &ev : events_) {
    PADDLE_ENFORCE(hipEventDestroy(ev.second));
  }
#endif
}

void OpHandleBase::Run(bool use_event) {
#ifdef PADDLE_WITH_CUDA
  if (events_.empty() && use_event) {
    for (auto &p : dev_ctxes_) {
      int dev_id = boost::get<platform::CUDAPlace>(p.first).device;
      PADDLE_ENFORCE(cudaSetDevice(dev_id));
      PADDLE_ENFORCE(
          cudaEventCreateWithFlags(&events_[dev_id], cudaEventDisableTiming));
    }
  }
#elif defined(PADDLE_WITH_HIP)
  if (events_.empty() && use_event) {
    for (auto &p : dev_ctxes_) {
      int dev_id = boost::get<platform::CUDAPlace>(p.first).device;
      PADDLE_ENFORCE(hipSetDevice(dev_id));
      PADDLE_ENFORCE(
          hipEventCreateWithFlags(&events_[dev_id], hipEventDisableTiming));
    }
  }
#else
  PADDLE_ENFORCE(!use_event);
#endif

  RunImpl();

#ifdef PADDLE_WITH_CUDA
  if (use_event) {
    for (auto &p : dev_ctxes_) {
      int dev_id = boost::get<platform::CUDAPlace>(p.first).device;
      auto stream =
          static_cast<platform::CUDADeviceContext *>(p.second)->stream();
      PADDLE_ENFORCE(cudaEventRecord(events_.at(dev_id), stream));
    }
  }
#elif defined(PADDLE_WITH_HIP)
  if (use_event) {
    for (auto &p : dev_ctxes_) {
      int dev_id = boost::get<platform::CUDAPlace>(p.first).device;
      auto stream =
          static_cast<platform::CUDADeviceContext *>(p.second)->stream();
      PADDLE_ENFORCE(hipEventRecord(events_.at(dev_id), stream));
    }
  }
#endif
}

void OpHandleBase::Wait(platform::DeviceContext *waited_dev) {
#ifdef PADDLE_WITH_CUDA
  if (platform::is_cpu_place(waited_dev->GetPlace()) || events_.empty()) {
    for (auto &dev_ctx : dev_ctxes_) {
      dev_ctx.second->Wait();
    }
  } else {
    auto stream =
        static_cast<platform::CUDADeviceContext *>(waited_dev)->stream();
    for (auto &ev : events_) {
      PADDLE_ENFORCE(cudaStreamWaitEvent(stream, ev.second, 0));
    }
  }
#elif defined(PADDLE_WITH_HIP)
  if (platform::is_cpu_place(waited_dev->GetPlace()) || events_.empty()) {
    for (auto &dev_ctx : dev_ctxes_) {
      dev_ctx.second->Wait();
    }
  } else {
    auto stream =
        static_cast<platform::CUDADeviceContext *>(waited_dev)->stream();
    for (auto &ev : events_) {
      PADDLE_ENFORCE(hipStreamWaitEvent(stream, ev.second, 0));
    }
  }
#else
  for (auto &dev_ctx : dev_ctxes_) {
    dev_ctx.second->Wait();
  }
#endif
}

void OpHandleBase::AddInput(VarHandleBase *in) {
  this->inputs_.emplace_back(in);
  in->pending_ops_.insert(this);
}

void OpHandleBase::AddOutput(VarHandleBase *out) {
  outputs_.emplace_back(out);
  out->generated_op_ = this;
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
