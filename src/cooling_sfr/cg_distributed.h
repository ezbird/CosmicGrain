#pragma once
// Building blocks for decomposition-independent feedback and dust searches.
//
// Feedback and dust operations act on everything within some radius of an
// event (a supernova, an AGB star, a forming star, a dust grain), but each
// MPI task only holds the particles of its own domain.  The pattern used
// throughout is:
//
//   1. allgather() the batch of events, so every task sees every event;
//   2. each task searches its own particles with a LocalTree;
//   3. per-event results go back to the event owner with exchange(), or are
//      summed over tasks with MPI_Allreduce;
//   4. the owner decides, and changes go to the particle owners with
//      exchange().
//
// The number of collectives per batch is independent of the number of
// events, and the neighbour set of an event does not depend on how the
// volume is split between tasks.

#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "../data/macros.h"
#include "../data/simparticles.h"
#include "rc1_gas_geometry.h"

namespace cg_dist
{
// Personalised all-to-all of trivially copyable records.  out[r] goes to task
// r; the result holds everything received, ordered by source task.  If
// source is given it receives the sending task of each record.
template <class T>
std::vector<T> exchange(const std::vector<std::vector<T>> &out, MPI_Comm comm, std::vector<int> *source = nullptr)
{
  static_assert(std::is_trivially_copyable<T>::value, "exchange() needs trivially copyable records");
  int ntask;
  MPI_Comm_size(comm, &ntask);
  if((int)out.size() != ntask)
    Terminate("cg_dist::exchange: %d send lists for %d tasks", (int)out.size(), ntask);

  std::vector<int> send_count(ntask), recv_count(ntask);
  for(int r = 0; r < ntask; r++)
    {
      if(out[r].size() > (size_t)(INT_MAX / sizeof(T)))
        Terminate("cg_dist::exchange: too many records for one task");
      send_count[r] = (int)out[r].size();
    }
  MPI_Alltoall(send_count.data(), 1, MPI_INT, recv_count.data(), 1, MPI_INT, comm);

  std::vector<int> send_bytes(ntask), recv_bytes(ntask), send_off(ntask), recv_off(ntask);
  long long send_total = 0, recv_total = 0;
  for(int r = 0; r < ntask; r++)
    {
      send_bytes[r] = (int)(send_count[r] * sizeof(T));
      recv_bytes[r] = (int)(recv_count[r] * sizeof(T));
      send_off[r]   = (int)send_total;
      recv_off[r]   = (int)recv_total;
      send_total += send_bytes[r];
      recv_total += recv_bytes[r];
      if(send_total > INT_MAX || recv_total > INT_MAX)
        Terminate("cg_dist::exchange: message exceeds MPI integer limits");
    }

  std::vector<T> send((size_t)(send_total / sizeof(T)) + 1), recv((size_t)(recv_total / sizeof(T)) + 1);
  for(int r = 0; r < ntask; r++)
    std::copy(out[r].begin(), out[r].end(), send.begin() + send_off[r] / sizeof(T));
  MPI_Alltoallv(send.data(), send_bytes.data(), send_off.data(), MPI_BYTE, recv.data(), recv_bytes.data(), recv_off.data(), MPI_BYTE,
                comm);
  recv.resize((size_t)(recv_total / sizeof(T)));

  if(source)
    {
      source->clear();
      source->reserve(recv.size());
      for(int r = 0; r < ntask; r++)
        source->insert(source->end(), recv_count[r], r);
    }
  return recv;
}

// Gather every task's records on every task, ordered by source task.
template <class T>
std::vector<T> allgather(const std::vector<T> &mine, MPI_Comm comm, std::vector<int> *source = nullptr)
{
  static_assert(std::is_trivially_copyable<T>::value, "allgather() needs trivially copyable records");
  int ntask;
  MPI_Comm_size(comm, &ntask);
  if(mine.size() > (size_t)(INT_MAX / sizeof(T)))
    Terminate("cg_dist::allgather: too many records");
  int n = (int)mine.size();
  std::vector<int> count(ntask), bytes(ntask), off(ntask);
  MPI_Allgather(&n, 1, MPI_INT, count.data(), 1, MPI_INT, comm);
  long long total = 0;
  for(int r = 0; r < ntask; r++)
    {
      bytes[r] = (int)(count[r] * sizeof(T));
      off[r]   = (int)total;
      total += bytes[r];
      if(total > INT_MAX)
        Terminate("cg_dist::allgather: message exceeds MPI integer limits");
    }
  std::vector<T> all((size_t)(total / sizeof(T)) + 1);
  T dummy{};
  MPI_Allgatherv(mine.empty() ? &dummy : mine.data(), (int)(n * sizeof(T)), MPI_BYTE, all.data(), bytes.data(), off.data(), MPI_BYTE,
                 comm);
  all.resize((size_t)(total / sizeof(T)));
  if(source)
    {
      source->clear();
      source->reserve(all.size());
      for(int r = 0; r < ntask; r++)
        source->insert(source->end(), count[r], r);
    }
  return all;
}

// Sum a vector of doubles over all tasks in place.
inline void sum_over_tasks(std::vector<double> &v, MPI_Comm comm)
{
  if(v.size() > (size_t)INT_MAX)
    Terminate("cg_dist::sum_over_tasks: vector too long");
  int n = (int)v.size(), nmax = 0;
  MPI_Allreduce(&n, &nmax, 1, MPI_INT, MPI_MAX, comm);
  if(nmax != n)
    Terminate("cg_dist::sum_over_tasks: tasks disagree on the vector length");
  if(n > 0)
    MPI_Allreduce(MPI_IN_PLACE, v.data(), n, MPI_DOUBLE, MPI_SUM, comm);
}

// Spatial index over a subset of this task's particles, using exact periodic
// integer-position distances.  Rebuilt for each batch, so it never holds
// stale indices.
class LocalTree
{
  using Geometry = cg_rc1_geometry::Tree<MyIntPosType>;
  Geometry tree;
  size_t count = 0;

  static double distance2(simparticles *Sp, const MyIntPosType *a, const MyIntPosType *b)
  {
    double d[3];
    Sp->nearest_image_intpos_to_pos(a, b, d);
    return d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
  }

 public:
  template <class Keep>
  void build(simparticles *Sp, int begin, int end, Keep keep)
  {
    std::vector<Geometry::Entry> entries;
    for(int i = begin; i < end; i++)
      if(keep(i))
        {
          Geometry::Entry e;
          e.slot = i;
          e.id   = (std::uint64_t)Sp->P[i].ID.get();
          for(int k = 0; k < 3; k++)
            e.x[k] = Sp->P[i].IntPos[k];
          entries.push_back(e);
        }
    count = entries.size();
    tree.build(std::move(entries));
  }

  // Index n arbitrary points, e.g. records gathered from other tasks;
  // position(k, x) fills x with the position of point k.  Searches then
  // report k in place of a particle index.
  template <class Position>
  void build_points(size_t n, Position position)
  {
    std::vector<Geometry::Entry> entries(n);
    for(size_t k = 0; k < n; k++)
      {
        entries[k].slot = (int)k;
        entries[k].id   = k;
        position(k, entries[k].x.data());
      }
    count = n;
    tree.build(std::move(entries));
  }

  size_t size() const { return count; }

  // visit(index, r2) for every member within radius of center.
  template <class Visit>
  void radius(simparticles *Sp, const MyIntPosType *center, double radius, Visit visit) const
  {
    if(count == 0 || !(radius > 0))
      return;
    auto metric = [Sp](const MyIntPosType *a, const MyIntPosType *b) { return distance2(Sp, a, b); };
    tree.radius(center, radius * radius, metric, [&](const Geometry::Entry &e, double d2) { visit(e.slot, d2); });
  }

  // Nearest member (ties broken by lower ID).  Returns false if empty.
  bool nearest(simparticles *Sp, const MyIntPosType *center, int &index, double &r2) const
  {
    if(count == 0)
      return false;
    auto metric = [Sp](const MyIntPosType *a, const MyIntPosType *b) { return distance2(Sp, a, b); };
    Geometry::Entry best;
    if(!tree.nearest(center, metric, [](const Geometry::Entry &) { return true; }, best, r2))
      return false;
    index = best.slot;
    return true;
  }
};

// Total gas + star + dust mass over all tasks, for conservation self-checks.
inline double total_baryon_mass(simparticles *Sp, MPI_Comm comm)
{
  long double local = 0;
  for(int i = 0; i < Sp->NumPart; i++)
    {
      const int type = Sp->P[i].getType();
      if(type == 0 || type == 4 || type == 6)
        local += Sp->P[i].getMass();
    }
  double mine = (double)local, total = 0;
  MPI_Allreduce(&mine, &total, 1, MPI_DOUBLE, MPI_SUM, comm);
  return total;
}

// Global nearest particle to each of a set of points.  Each task supplies
// its local nearest (r2 = +inf when it has none); the result keeps the
// smallest r2, ties broken by lower ID, so it does not depend on the
// decomposition.  The payload travels with the winner.
struct NearestRecord
{
  double r2;
  std::uint64_t id;
  int task;
  int index;
  double payload[4];
};

inline void nearest_reduce_op(void *in, void *inout, int *len, MPI_Datatype *)
{
  const NearestRecord *a = static_cast<const NearestRecord *>(in);
  NearestRecord *b       = static_cast<NearestRecord *>(inout);
  for(int i = 0; i < *len; i++)
    if(a[i].r2 < b[i].r2 || (a[i].r2 == b[i].r2 && a[i].id < b[i].id))
      b[i] = a[i];
}

inline void nearest_over_tasks(std::vector<NearestRecord> &v, MPI_Comm comm)
{
  static MPI_Datatype type = MPI_DATATYPE_NULL;
  static MPI_Op op         = MPI_OP_NULL;
  if(type == MPI_DATATYPE_NULL)
    {
      MPI_Type_contiguous(sizeof(NearestRecord), MPI_BYTE, &type);
      MPI_Type_commit(&type);
      MPI_Op_create(nearest_reduce_op, 1, &op);
    }
  if(v.size() > (size_t)INT_MAX)
    Terminate("cg_dist::nearest_over_tasks: vector too long");
  if(!v.empty())
    MPI_Allreduce(MPI_IN_PLACE, v.data(), (int)v.size(), type, op, comm);
}

}  // namespace cg_dist
