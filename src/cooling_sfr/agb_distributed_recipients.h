#pragma once
// Collective reference selector, not yet integrated into GADGET's tree walk.
#include <mpi.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <climits>
#include <stdexcept>
namespace cg_agb_distributed {
struct Candidate {std::uint64_t id;double r2,mass;int owner;};
inline bool less(const Candidate &a,const Candidate &b){return a.r2<b.r2 || (a.r2==b.r2 && a.id<b.id);}
inline void require_all(bool ok,MPI_Comm comm,const char *why){int bad=ok?0:1,global=0;MPI_Allreduce(&bad,&global,1,MPI_INT,MPI_MAX,comm);if(global)throw std::runtime_error(why);}
inline std::vector<Candidate> select(std::vector<Candidate> local,int cap,MPI_Comm comm){
 int rank,size;MPI_Comm_rank(comm,&rank);MPI_Comm_size(comm,&size);
 int lo,hi;MPI_Allreduce(&cap,&lo,1,MPI_INT,MPI_MIN,comm);MPI_Allreduce(&cap,&hi,1,MPI_INT,MPI_MAX,comm);
 bool valid=lo==hi && cap>0 && cap<=65536;
 for(auto &p:local){valid=valid && std::isfinite(p.r2) && p.r2>=0 && std::isfinite(p.mass) && p.mass>0;p.owner=rank;}
 require_all(valid,comm,"Invalid candidate or inconsistent recipient cap");
 // Preserve the exact sorted nearest-K prefix without sorting the discarded tail.
 if(local.size()>(size_t)cap)
   std::partial_sort(local.begin(),local.begin()+cap,local.end(),less);
 else
   std::sort(local.begin(),local.end(),less);
 if(local.size()>(size_t)cap)local.resize(cap);
 // Global nearest K is contained in the union of each rank's nearest K.
 int n=(int)local.size();std::vector<int> counts(size),offsets(size);MPI_Allgather(&n,1,MPI_INT,counts.data(),1,MPI_INT,comm);
 long long total64=0;for(int r=0;r<size;++r){offsets[r]=(int)total64;total64+=counts[r];if(total64>INT_MAX)break;}
 require_all(total64<=INT_MAX,comm,"Recipient exchange exceeds MPI count range");
 int total=(int)total64;
 std::vector<std::uint64_t> id(n),ids(total);std::vector<double> d(n),m(n),ds(total),ms(total);
 for(int i=0;i<n;++i){id[i]=local[i].id;d[i]=local[i].r2;m[i]=local[i].mass;}
 MPI_Allgatherv(id.data(),n,MPI_UINT64_T,ids.data(),counts.data(),offsets.data(),MPI_UINT64_T,comm);
 MPI_Allgatherv(d.data(),n,MPI_DOUBLE,ds.data(),counts.data(),offsets.data(),MPI_DOUBLE,comm);
 MPI_Allgatherv(m.data(),n,MPI_DOUBLE,ms.data(),counts.data(),offsets.data(),MPI_DOUBLE,comm);
 std::vector<Candidate> result;result.reserve(total);
 for(int r=0;r<size;++r)for(int i=offsets[r];i<offsets[r]+counts[r];++i)result.push_back({ids[i],ds[i],ms[i],r});
 std::vector<std::uint64_t> unique=ids;std::sort(unique.begin(),unique.end());
 require_all(std::adjacent_find(unique.begin(),unique.end())==unique.end(),comm,"Duplicate gas IDs in candidate exchange");
 std::sort(result.begin(),result.end(),less);if(result.size()>(size_t)cap)result.resize(cap);return result;
}
}
