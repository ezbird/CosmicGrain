#pragma once
#include <cstdint>
#include <unordered_map>
#include <stdexcept>

// Ephemeral ID-to-array-index acceleration only. Never caches physical state.
// Gas IDs must be unique, as required by the distributed recipient selector.
namespace cg_rc1_gas_index {
class Index {
  std::unordered_map<std::uint64_t,int> indices;
public:
  template<class Id,class Type>
  void build(int n,Id id,Type type) {
    indices.clear();indices.reserve(n);
    for(int j=0;j<n;++j)if(type(j)==0)
      if(!indices.emplace(id(j),j).second)
        throw std::runtime_error("Duplicate local gas ID in AGB index");
  }
  template<class Id,class Type>
  int find(std::uint64_t wanted,int n,Id id,Type type) {
    auto it=indices.find(wanted);
    if(it!=indices.end()) {
      const int j=it->second;
      if(j>=0 && j<n && type(j)==0 && id(j)==wanted)return j;
    }
    // Handle new particles, deletion, array permutation, and changed types.
    // Never retain an unvalidated array index across particle operations.
    for(int j=0;j<n;++j)if(type(j)==0 && id(j)==wanted) {
      indices[wanted]=j;return j;
    }
    indices.erase(wanted);return -1;
  }
};
}
