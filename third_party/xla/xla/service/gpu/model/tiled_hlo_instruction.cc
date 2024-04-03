/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/service/gpu/model/tiled_hlo_instruction.h"

#include <cstddef>
#include <sstream>
#include <string>

#include "absl/hash/hash.h"
#include "absl/strings/str_join.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/model/indexing_map.h"

namespace xla {
namespace gpu {

size_t TiledHloInstruction::PtrHash::operator()(
    const TiledHloInstruction* tiled_hlo) const {
  return absl::HashOf(*tiled_hlo);
}

bool TiledHloInstruction::PtrEqual::operator()(
    const TiledHloInstruction* lhs, const TiledHloInstruction* rhs) const {
  return *lhs == *rhs;
}

bool operator==(const TiledHloInstruction& lhs,
                const TiledHloInstruction& rhs) {
  return lhs.hlo() == rhs.hlo() && lhs.tile_sizes() == rhs.tile_sizes() &&
         lhs.tile_strides() == rhs.tile_strides() &&
         lhs.block_id_to_tile_offsets_indexing() ==
             rhs.block_id_to_tile_offsets_indexing();
}

std::string TiledHloInstruction::ToString() const {
  std::stringstream ss;
  ss << "hlo: " << hlo_->ToString() << "\n";
  ss << "tile_sizes: {" << absl::StrJoin(tile_sizes_, ", ") << "}\n";
  ss << "tile_strides: {" << absl::StrJoin(tile_strides_, ", ") << "}\n";
  ss << "block_id_to_tile_offsets_indexing: "
     << block_id_to_tile_offsets_indexing_;
  return ss.str();
}

}  // namespace gpu
}  // namespace xla
