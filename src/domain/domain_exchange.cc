/*******************************************************************************
 * \copyright   This file is part of the GADGET4 N-body/SPH code developed
 * \copyright   by Volker Springel. Copyright (C) 2014-2020 by Volker Springel
 * \copyright   (vspringel@mpa-garching.mpg.de) and all contributing authors.
 *******************************************************************************/

/*! \file domain_exchange.cc
 *
 *  \brief routines for moving particle data between MPI ranks
 */

#include "gadgetconfig.h"

#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "../data/allvars.h"
#include "../data/dtypes.h"
#include "../data/mymalloc.h"
#include "../domain/domain.h"
#include "../fof/fof.h"
#include "../logs/timer.h"
#include "../main/simulation.h"
#include "../mpi_utils/mpi_utils.h"
#include "../ngbtree/ngbtree.h"
#include "../sort/cxxsort.h"
#include "../system/system.h"

#ifdef DUST
#include "../dust/dust.h"
#endif

/*! \file domain_exchange.c
 *  \brief exchanges particle data according to the new domain decomposition
 */
template <typename partset>
void domain<partset>::domain_resize_storage(int count_get_total, int count_get_sph, int option_flag)
{
  int max_load, load       = count_get_total;
  int max_sphload, sphload = count_get_sph;
  MPI_Allreduce(&load, &max_load, 1, MPI_INT, MPI_MAX, Communicator);
  MPI_Allreduce(&sphload, &max_sphload, 1, MPI_INT, MPI_MAX, Communicator);

  if(max_load > (1.0 - ALLOC_TOLERANCE) * Tp->MaxPart || max_load < (1.0 - 3 * ALLOC_TOLERANCE) * Tp->MaxPart)
    {
      Tp->reallocate_memory_maxpart(max_load / (1.0 - 2 * ALLOC_TOLERANCE));

      if(option_flag == 1)
        domain_key = (peanokey *)Mem.myrealloc_movable(domain_key, sizeof(peanokey) * Tp->MaxPart);
    }

  if(max_sphload > (1.0 - ALLOC_TOLERANCE) * Tp->MaxPartSph || max_sphload < (1.0 - 3 * ALLOC_TOLERANCE) * Tp->MaxPartSph)
    {
      int maxpartsphNew = max_sphload / (1.0 - 2 * ALLOC_TOLERANCE);
      if(option_flag == 2)
        {
          Terminate("need to reactivate this");
        }
      Tp->reallocate_memory_maxpartsph(maxpartsphNew);
    }
}

/*! This function determines how many particles that are currently stored
 *  on the local CPU have to be moved off according to the domain
 *  decomposition.
 */
template <typename partset>
void domain<partset>::domain_countToGo(int *toGoDM, int *toGoSph)
{
  for(int n = 0; n < NTask; n++)
    {
      toGoDM[n] = toGoSph[n] = 0;
    }

  for(int n = 0; n < Tp->NumPart; n++)
    {
      int no = n_to_no(n);

      if(Tp->P[n].getType() == 0)
        toGoSph[TaskOfLeaf[no]]++;
      else
        toGoDM[TaskOfLeaf[no]]++;
    }
}

#ifdef DUST
/*! Count how many dust particles need to be moved 
 *  CRITICAL FIX: Newly created particles may not be in tree yet!
 */
template <typename partset>
void domain<partset>::domain_countToGo_dust(int *toGoDust)
{
  for(int n = 0; n < NTask; n++)
    toGoDust[n] = 0;

  for(int n = 0; n < Tp->NumPart; n++)
    {
      if(Tp->P[n].getType() == 6)  // DUST_PARTICLE_TYPE
        {
          // CRITICAL: Check if particle is in the domain tree
          // Newly created particles may not have valid tree indices yet
          
          // Method 1: Check if particle was integrated this timestep
          // Particles created this timestep haven't been through tree construction
          if(Tp->P[n].Ti_Current != All.Ti_Current) {
            // Fresh particle - keep on local task
            toGoDust[ThisTask]++;
            continue;
          }
          
          // Method 2: Bounds check on tree node
          int no = n_to_no(n);
          
          // Sanity check: valid tree node (negative means invalid)
          if(no < 0) {
            // Invalid tree node - keep on local task
            toGoDust[ThisTask]++;
            continue;
          }
          
          // Additional safety: check task is valid
          int task = TaskOfLeaf[no];
          if(task < 0 || task >= NTask) {
            // Invalid task - keep on local task
            toGoDust[ThisTask]++;
            continue;
          }
          
          // Particle is in tree, use normal decomposition
          toGoDust[task]++;
        }
    }
}
#endif

template <typename partset>
void domain<partset>::domain_coll_subfind_prepare_exchange(void)
{
#ifdef SUBFIND
  for(int i = 0; i < Tp->NumPart; i++)
    {
      int task = TaskOfLeaf[n_to_no(i)];

      Tp->PS[i].TargetTask  = task;
      Tp->PS[i].TargetIndex = 0; /* unimportant here */
    }
#endif
}



/* Added pieces for the dust array (DustP). Gadget uses a stack-based memory allocator that requires LIFO (last-in-first-out) 
deallocation... so carefully deallocate in reverse order. */
template <typename partset>
void domain<partset>::domain_exchange(void)
{
  double t0 = Logs.second();

  int *toGoDM   = (int *)Mem.mymalloc_movable(&toGoDM, "toGoDM", NTask * sizeof(int));
  int *toGoSph  = (int *)Mem.mymalloc_movable(&toGoSph, "toGoSph", NTask * sizeof(int));
  int *toGetDM  = (int *)Mem.mymalloc_movable(&toGetDM, "toGetDM", NTask * sizeof(int));
  int *toGetSph = (int *)Mem.mymalloc_movable(&toGetSph, "toGetSph", NTask * sizeof(int));

  domain_countToGo(toGoDM, toGoSph);

#ifdef DUST
  int *toGoDust = (int *)Mem.mymalloc_movable(&toGoDust, "toGoDust", NTask * sizeof(int));
  int *toGetDust = (int *)Mem.mymalloc_movable(&toGetDust, "toGetDust", NTask * sizeof(int));
  domain_countToGo_dust(toGoDust);
#endif

  int *toGo  = (int *)Mem.mymalloc("toGo", 2 * NTask * sizeof(int));
  int *toGet = (int *)Mem.mymalloc("toGet", 2 * NTask * sizeof(int));

  for(int i = 0; i < NTask; ++i)
    {
      toGo[2 * i]     = toGoDM[i];
      toGo[2 * i + 1] = toGoSph[i];
    }
  myMPI_Alltoall(toGo, 2, MPI_INT, toGet, 2, MPI_INT, Communicator);
  for(int i = 0; i < NTask; ++i)
    {
      toGetDM[i]  = toGet[2 * i];
      toGetSph[i] = toGet[2 * i + 1];
    }
  Mem.myfree(toGet);
  Mem.myfree(toGo);

#ifdef DUST
  // Exchange dust counts
  myMPI_Alltoall(toGoDust, 1, MPI_INT, toGetDust, 1, MPI_INT, Communicator);
#endif

  int count_togo_dm = 0, count_togo_sph = 0, count_get_dm = 0, count_get_sph = 0;
  for(int i = 0; i < NTask; i++)
    {
      count_togo_dm += toGoDM[i];
      count_togo_sph += toGoSph[i];
      count_get_dm += toGetDM[i];
      count_get_sph += toGetSph[i];
    }

#ifdef DUST
  int count_togo_dust = 0, count_get_dust = 0;
  for(int i = 0; i < NTask; i++)
    {
      count_togo_dust += toGoDust[i];
      count_get_dust += toGetDust[i];
    }
#endif

  long long sumtogo = count_togo_dm;
  sumup_longs(1, &sumtogo, &sumtogo, Communicator);

  domain_printf("DOMAIN: exchange of %lld particles\n", sumtogo);

  if(Tp->NumPart != count_togo_dm + count_togo_sph)
    Terminate("NumPart != count_togo");

  int *send_sph_offset = (int *)Mem.mymalloc_movable(&send_sph_offset, "send_sph_offset", NTask * sizeof(int));
  int *send_dm_offset  = (int *)Mem.mymalloc_movable(&send_dm_offset, "send_dm_offset", NTask * sizeof(int));
  int *recv_sph_offset = (int *)Mem.mymalloc_movable(&recv_sph_offset, "recv_sph_offset", NTask * sizeof(int));
  int *recv_dm_offset  = (int *)Mem.mymalloc_movable(&recv_dm_offset, "recv_dm_offset", NTask * sizeof(int));

#ifdef DUST
  int *send_dust_offset = (int *)Mem.mymalloc_movable(&send_dust_offset, "send_dust_offset", NTask * sizeof(int));
  int *recv_dust_offset = (int *)Mem.mymalloc_movable(&recv_dust_offset, "recv_dust_offset", NTask * sizeof(int));
#endif

  send_sph_offset[0] = send_dm_offset[0] = recv_sph_offset[0] = recv_dm_offset[0] = 0;
#ifdef DUST
  send_dust_offset[0] = recv_dust_offset[0] = 0;
#endif

  for(int i = 1; i < NTask; i++)
    {
      send_sph_offset[i] = send_sph_offset[i - 1] + toGoSph[i - 1];
      send_dm_offset[i]  = send_dm_offset[i - 1] + toGoDM[i - 1];

      recv_sph_offset[i] = recv_sph_offset[i - 1] + toGetSph[i - 1];
      recv_dm_offset[i]  = recv_dm_offset[i - 1] + toGetDM[i - 1];

#ifdef DUST
      send_dust_offset[i] = send_dust_offset[i - 1] + toGoDust[i - 1];
      recv_dust_offset[i] = recv_dust_offset[i - 1] + toGetDust[i - 1];
#endif
    }

  for(int i = 0; i < NTask; i++)
    {
      send_dm_offset[i] += count_togo_sph;
      recv_dm_offset[i] += count_get_sph;
    }

  pdata *partBuf =
      (typename partset::pdata *)Mem.mymalloc_movable_clear(&partBuf, "partBuf", (count_togo_dm + count_togo_sph) * sizeof(pdata));
  sph_particle_data *sphBuf =
      (sph_particle_data *)Mem.mymalloc_movable_clear(&sphBuf, "sphBuf", count_togo_sph * sizeof(sph_particle_data));
  peanokey *keyBuf = (peanokey *)Mem.mymalloc_movable_clear(&keyBuf, "keyBuf", (count_togo_dm + count_togo_sph) * sizeof(peanokey));

#ifdef DUST
  dust_data *dustBuf = NULL;
  if(count_togo_dust > 0)
    dustBuf = (dust_data *)Mem.mymalloc_movable_clear(&dustBuf, "dustBuf", count_togo_dust * sizeof(dust_data));
#endif

  for(int i = 0; i < NTask; i++)
    {
      toGoSph[i] = toGoDM[i] = 0;
#ifdef DUST
      toGoDust[i] = 0;
#endif
    }

for(int n = 0; n < Tp->NumPart; n++)
  {
    int off, num;
    
    // CRITICAL: Use same logic as domain_countToGo for task assignment
    int task;
    
    // Check if particle is in tree (same checks as counting phase)
    if(Tp->P[n].Ti_Current != All.Ti_Current) {
      // Fresh particle - keep on local task
      task = ThisTask;
    } else {
      int no = n_to_no(n);
      if(no < 0) {
        // Invalid tree node - keep on local task
        task = ThisTask;
      } else {
        // Get task from tree
        task = TaskOfLeaf[no];
        // Additional safety check
        if(task < 0 || task >= NTask) {
          task = ThisTask;
        }
      }
    }

    if(Tp->P[n].getType() == 0)
      {
        num = toGoSph[task]++;
        off         = send_sph_offset[task] + num;
        sphBuf[off] = Tp->SphP[n];
      }
    else
      {
        num = toGoDM[task]++;
        off = send_dm_offset[task] + num;

  #ifdef DUST
        // If this is a dust particle, also pack DustP
        if(Tp->P[n].getType() == 6)
          {
            int dust_num = toGoDust[task]++;
            int dust_off = send_dust_offset[task] + dust_num;
            dustBuf[dust_off] = Tp->DustP[n];
          }
  #endif
      }

    partBuf[off] = Tp->P[n];
    keyBuf[off]  = domain_key[n];
  }

  /**** now resize the storage for the P[] and SphP[] arrays if needed ****/
  domain_resize_storage(count_get_dm + count_get_sph, count_get_sph, 1);

  /*****  space has been created, now we can do the actual exchange *****/

  /* produce a flag if any of the send sizes is above our transfer limit, in this case we will
   * transfer the data in chunks.
   */

  int flag_big = 0, flag_big_all;
  for(int i = 0; i < NTask; i++)
    {
      if(toGoSph[i] * sizeof(sph_particle_data) > MPI_MESSAGE_SIZELIMIT_IN_BYTES)
        flag_big = 1;

      if(std::max<int>(toGoSph[i], toGoDM[i]) * sizeof(typename partset::pdata) > MPI_MESSAGE_SIZELIMIT_IN_BYTES)
        flag_big = 1;

#ifdef DUST
      if(toGoDust[i] * sizeof(dust_data) > MPI_MESSAGE_SIZELIMIT_IN_BYTES)
        flag_big = 1;
#endif
    }

  MPI_Allreduce(&flag_big, &flag_big_all, 1, MPI_INT, MPI_MAX, Communicator);

#if 1
#ifdef USE_MPIALLTOALLV_IN_DOMAINDECOMP
  int method = 0;
#else
#ifndef ISEND_IRECV_IN_DOMAIN /* synchronous communication */
  int method = 1;
#else
  int method = 2; /* asynchronous communication */
#endif
#endif
  MPI_Datatype tp;
  MPI_Type_contiguous(sizeof(typename partset::pdata), MPI_CHAR, &tp);
  MPI_Type_commit(&tp);
  myMPI_Alltoallv_new(partBuf, toGoSph, send_sph_offset, tp, Tp->P, toGetSph, recv_sph_offset, tp, Communicator, method);
  myMPI_Alltoallv_new(partBuf, toGoDM, send_dm_offset, tp, Tp->P, toGetDM, recv_dm_offset, tp, Communicator, method);
  MPI_Type_free(&tp);
  
  MPI_Type_contiguous(sizeof(sph_particle_data), MPI_CHAR, &tp);
  MPI_Type_commit(&tp);
  myMPI_Alltoallv_new(sphBuf, toGoSph, send_sph_offset, tp, Tp->SphP, toGetSph, recv_sph_offset, tp, Communicator, method);
  MPI_Type_free(&tp);

#ifdef DUST
  // Receive dust data into temporary buffer (not directly into DustP)
  dust_data *dustBuf_recv = NULL;
  if(count_get_dust > 0)
    dustBuf_recv = (dust_data *)Mem.mymalloc("dustBuf_recv", count_get_dust * sizeof(dust_data));
  
  if(count_togo_dust > 0 || count_get_dust > 0)
    {
      MPI_Type_contiguous(sizeof(dust_data), MPI_CHAR, &tp);
      MPI_Type_commit(&tp);
      myMPI_Alltoallv_new(dustBuf, toGoDust, send_dust_offset, tp, dustBuf_recv, toGetDust, recv_dust_offset, tp, Communicator, method);
      MPI_Type_free(&tp);
    }

  // CRITICAL: Zero DustP for ALL gas particles FIRST (they should never have dust properties)
  for(int i = 0; i < count_get_sph; i++)
    memset(&Tp->DustP[i], 0, sizeof(dust_data));

  // NOW copy DustP data to correct indices (AFTER all P[] exchanges are complete!)
  if(count_get_dust > 0)
    {
      int dust_recv_idx = 0;
      for(int i = count_get_sph; i < count_get_sph + count_get_dm; i++)
        {
          if(Tp->P[i].getType() == 6)
            {
              Tp->DustP[i] = dustBuf_recv[dust_recv_idx];
              dust_recv_idx++;
            }
          else
            {
              // CRITICAL: Zero out DustP for non-dust particles!
              memset(&Tp->DustP[i], 0, sizeof(dust_data));
            }
        }
      
      if(dust_recv_idx != count_get_dust)
        Terminate("DUST: Received %d dust particles but expected %d", dust_recv_idx, count_get_dust);
      
      Mem.myfree(dustBuf_recv);
    }
  else
    {
      // No dust received - zero ALL DustP entries for DM particles
      for(int i = count_get_sph; i < count_get_sph + count_get_dm; i++)
        memset(&Tp->DustP[i], 0, sizeof(dust_data));
    }
#endif

  MPI_Type_contiguous(sizeof(peanokey), MPI_CHAR, &tp);
  MPI_Type_commit(&tp);
  myMPI_Alltoallv_new(keyBuf, toGoSph, send_sph_offset, tp, domain_key, toGetSph, recv_sph_offset, tp, Communicator, method);
  myMPI_Alltoallv_new(keyBuf, toGoDM, send_dm_offset, tp, domain_key, toGetDM, recv_dm_offset, tp, Communicator, method);
  MPI_Type_free(&tp);


#else
  my_int_MPI_Alltoallv(partBuf, toGoSph, send_sph_offset, Tp->P, toGetSph, recv_sph_offset, sizeof(pdata), flag_big_all, Communicator);

  my_int_MPI_Alltoallv(sphBuf, toGoSph, send_sph_offset, Tp->SphP, toGetSph, recv_sph_offset, sizeof(sph_particle_data), flag_big_all,
                       Communicator);

#ifdef DUST
  // Receive dust data into temporary buffer
  dust_data *dustBuf_recv = NULL;
  if(count_get_dust > 0)
    dustBuf_recv = (dust_data *)Mem.mymalloc("dustBuf_recv", count_get_dust * sizeof(dust_data));
    
  if(count_togo_dust > 0 || count_get_dust > 0)
    my_int_MPI_Alltoallv(dustBuf, toGoDust, send_dust_offset, dustBuf_recv, toGetDust, recv_dust_offset, sizeof(dust_data), flag_big_all,
                         Communicator);
#endif

  my_int_MPI_Alltoallv(keyBuf, toGoSph, send_sph_offset, domain_key, toGetSph, recv_sph_offset, sizeof(peanokey), flag_big_all,
                       Communicator);

  my_int_MPI_Alltoallv(partBuf, toGoDM, send_dm_offset, Tp->P, toGetDM, recv_dm_offset, sizeof(pdata), flag_big_all, Communicator);

  my_int_MPI_Alltoallv(keyBuf, toGoDM, send_dm_offset, domain_key, toGetDM, recv_dm_offset, sizeof(peanokey), flag_big_all,
                       Communicator);

#ifdef DUST
// CRITICAL: Zero DustP for ALL gas particles (they should never have dust properties)
for(int i = 0; i < count_get_sph; i++)
  memset(&Tp->DustP[i], 0, sizeof(dust_data));

// NOW copy DustP data to correct indices (AFTER all P[] exchanges are complete!)
if(count_get_dust > 0)
  {
    int dust_recv_idx = 0;
    for(int i = count_get_sph; i < count_get_sph + count_get_dm; i++)
      {
        if(Tp->P[i].getType() == 6)
          {
            Tp->DustP[i] = dustBuf_recv[dust_recv_idx];
            dust_recv_idx++;
          }
        else
          {
            // CRITICAL: Zero out DustP for non-dust particles!
            memset(&Tp->DustP[i], 0, sizeof(dust_data));
          }
      }
      
      if(dust_recv_idx != count_get_dust)
        Terminate("DUST: Received %d dust particles but expected %d", dust_recv_idx, count_get_dust);
      
      Mem.myfree(dustBuf_recv);
    }
  else
    {
      // No dust received - zero ALL DustP entries for DM particles
      for(int i = count_get_sph; i < count_get_sph + count_get_dm; i++)
        memset(&Tp->DustP[i], 0, sizeof(dust_data));
    }
#endif

#endif

  Tp->NumPart = count_get_dm + count_get_sph;
  Tp->NumGas  = count_get_sph;

  // Free in REVERSE order of allocation (LIFO for stack-based allocator)
#ifdef DUST
  if(dustBuf)
    Mem.myfree(dustBuf);      // dustBuf allocated last, so free first
  // dustBuf_recv already freed above
#endif

  Mem.myfree(keyBuf);
  Mem.myfree(sphBuf);
  Mem.myfree(partBuf);

#ifdef DUST
  Mem.myfree(recv_dust_offset);
  Mem.myfree(send_dust_offset);
#endif

  Mem.myfree(recv_dm_offset);
  Mem.myfree(recv_sph_offset);
  Mem.myfree(send_dm_offset);
  Mem.myfree(send_sph_offset);

#ifdef DUST
  Mem.myfree(toGetDust);
  Mem.myfree(toGoDust);
#endif

  Mem.myfree(toGetSph);
  Mem.myfree(toGetDM);
  Mem.myfree(toGoSph);
  Mem.myfree(toGoDM);

  double t1 = Logs.second();

  domain_printf("DOMAIN: particle exchange done. (took %g sec)\n", Logs.timediff(t0, t1));
}

template <typename partset>
void domain<partset>::peano_hilbert_order(peanokey *key)
{
  mpi_printf("PEANO: Begin Peano-Hilbert order...\n");
  double t0 = Logs.second();

  if(Tp->NumGas)
    {
      peano_hilbert_data *pmp = (peano_hilbert_data *)Mem.mymalloc("pmp", sizeof(peano_hilbert_data) * Tp->NumGas);
      int *Id                 = (int *)Mem.mymalloc("Id", sizeof(int) * Tp->NumGas);

      for(int i = 0; i < Tp->NumGas; i++)
        {
          pmp[i].index = i;
          pmp[i].key   = key[i];
        }

      mycxxsort(pmp, pmp + Tp->NumGas, compare_peano_hilbert_data);

      for(int i = 0; i < Tp->NumGas; i++)
        Id[pmp[i].index] = i;

      reorder_gas(Id);

      Mem.myfree(Id);
      Mem.myfree(pmp);
    }

  if(Tp->NumPart - Tp->NumGas > 0)
    {
      peano_hilbert_data *pmp = (peano_hilbert_data *)Mem.mymalloc("pmp", sizeof(peano_hilbert_data) * (Tp->NumPart - Tp->NumGas));
      int *Id                 = (int *)Mem.mymalloc("Id", sizeof(int) * (Tp->NumPart - Tp->NumGas));

      for(int i = Tp->NumGas; i < Tp->NumPart; i++)
        {
          pmp[i - Tp->NumGas].index = i;
          pmp[i - Tp->NumGas].key   = key[i];
        }

      mycxxsort(pmp, pmp + Tp->NumPart - Tp->NumGas, compare_peano_hilbert_data);

      for(int i = Tp->NumGas; i < Tp->NumPart; i++)
        Id[pmp[i - Tp->NumGas].index - Tp->NumGas] = i;

      reorder_particles(Id - Tp->NumGas, Tp->NumGas, Tp->NumPart);

      Mem.myfree(Id);
      Mem.myfree(pmp);
    }

  mpi_printf("PEANO: done, took %g sec.\n", Logs.timediff(t0, Logs.second()));
}

template <typename partset>
void domain<partset>::reorder_gas(int *Id)
{
  for(int i = 0; i < Tp->NumGas; i++)
    {
      if(Id[i] != i)
        {
          pdata Psource                = Tp->P[i];
          sph_particle_data SphPsource = Tp->SphP[i];

          int idsource = Id[i];
          int dest     = Id[i];

          do
            {
              pdata Psave                = Tp->P[dest];
              sph_particle_data SphPsave = Tp->SphP[dest];
              int idsave                 = Id[dest];

              Tp->P[dest]    = Psource;
              Tp->SphP[dest] = SphPsource;
              Id[dest]       = idsource;

              if(dest == i)
                break;

              Psource    = Psave;
              SphPsource = SphPsave;
              idsource   = idsave;

              dest = idsource;
            }
          while(1);
        }
    }
}

/*! \brief Reorder non-gas particles according to a permutation array
 *
 *  This function reorders particles from Nstart to N according to the
 *  permutation given in Id[]. For dust particles (type 6), we also need
 *  to reorder the DustP[] array to keep grain properties (radius, temperature,
 *  etc.) synchronized with their parent P[] entries. This prevents corruption
 *  during domain decomposition when particles are reshuffled.
 *
 *  \param Id Permutation array indicating where each particle should go
 *  \param Nstart Starting index (typically Tp->NumGas for non-gas particles)
 *  \param N Ending index (typically Tp->NumPart)
 */
template <typename partset>
void domain<partset>::reorder_particles(int *Id, int Nstart, int N)
{
  for(int i = Nstart; i < N; i++)
    {
      if(Id[i] != i)
        {
          pdata Psource = Tp->P[i];
          
          #ifdef DUST
          auto DustPsource = Tp->DustP[i];
          #endif
          
          int idsource  = Id[i];
          int dest = Id[i];

          do
            {
              pdata Psave = Tp->P[dest];
              
              #ifdef DUST
              auto DustPsave = Tp->DustP[dest];
              #endif
              
              int idsave  = Id[dest];

              Tp->P[dest] = Psource;
              
              #ifdef DUST
              if(Psource.getType() == 6)
                Tp->DustP[dest] = DustPsource;
              else
                memset(&Tp->DustP[dest], 0, sizeof(dust_data));
              #endif
              
              Id[dest] = idsource;

              if(dest == i)
                break;

              Psource  = Psave;
              
              #ifdef DUST
              DustPsource = DustPsave;
              #endif
              
              idsource = idsave;
              dest = idsource;
            }
          while(1);
        }
    }
}

template <typename partset>
void domain<partset>::reorder_PS(int *Id, int Nstart, int N)
{
  for(int i = Nstart; i < N; i++)
    {
      if(Id[i] != i)
        {
          subfind_data PSsource = Tp->PS[i];

          int idsource = Id[i];
          int dest     = Id[i];

          do
            {
              subfind_data PSsave = Tp->PS[dest];
              int idsave          = Id[dest];

              Tp->PS[dest] = PSsource;
              Id[dest]     = idsource;

              if(dest == i)
                break;

              PSsource = PSsave;
              idsource = idsave;

              dest = idsource;
            }
          while(1);
        }
    }
}

template <typename partset>
void domain<partset>::reorder_P_and_PS(int *Id)
{
  for(int i = 0; i < Tp->NumPart; i++)
    {
      if(Id[i] != i)
        {
          pdata        Psource  = Tp->P[i];
          subfind_data PSsource = Tp->PS[i];

#ifdef DUST
          dust_data DustPsource = Tp->DustP[i];
#endif

          int idsource = Id[i];
          int dest     = Id[i];

          do
            {
              pdata        Psave  = Tp->P[dest];
              subfind_data PSsave = Tp->PS[dest];
              int          idsave = Id[dest];

#ifdef DUST
              dust_data DustPsave = Tp->DustP[dest];
#endif

              Tp->P[dest]  = Psource;
              Tp->PS[dest] = PSsource;

#ifdef DUST
              // Keep DustP aligned with the particle that now sits at dest
              if(Psource.getType() == 6)
                Tp->DustP[dest] = DustPsource;
              else
                memset(&Tp->DustP[dest], 0, sizeof(dust_data));
#endif

              Id[dest] = idsource;

              if(dest == i)
                break;

              Psource  = Psave;
              PSsource = PSsave;
              idsource = idsave;
              dest     = idsource;

#ifdef DUST
              DustPsource = DustPsave;
#endif
            }
          while(1);
        }
    }
}


template <typename partset>
void domain<partset>::reorder_P_PS(int loc_numgas, int loc_numpart)
{
  local_sort_data *mp = (local_sort_data *)Mem.mymalloc("mp", sizeof(local_sort_data) * (loc_numpart - loc_numgas));
  mp -= loc_numgas;

  int *Id = (int *)Mem.mymalloc("Id", sizeof(int) * (loc_numpart - loc_numgas));
  Id -= loc_numgas;

  for(int i = loc_numgas; i < loc_numpart; i++)
    {
      mp[i].index       = i;
      mp[i].targetindex = Tp->PS[i].TargetIndex;
    }

  mycxxsort(mp + loc_numgas, mp + loc_numpart, compare_local_sort_data_targetindex);

  for(int i = loc_numgas; i < loc_numpart; i++)
    Id[mp[i].index] = i;

  reorder_particles(Id, loc_numgas, loc_numpart);

  for(int i = loc_numgas; i < loc_numpart; i++)
    Id[mp[i].index] = i;

  reorder_PS(Id, loc_numgas, loc_numpart);

  Id += loc_numgas;
  Mem.myfree(Id);
  mp += loc_numgas;
  Mem.myfree(mp);
}

/* This function redistributes the particles according to what is stored in
 * PS[].TargetTask, and PS[].TargetIndex.
 *
 * DUST handling:
 *  - DustP[] is a sidecar array indexed like P[].
 *  - When exchanging P[], we also exchange a dust buffer with identical layout.
 *  - For non-dust particles, we send/receive a zero dust_data struct.
 *
 * IMPORTANT: mymalloc stack allocator requires strict LIFO frees.
 */
template <typename partset>
void domain<partset>::particle_exchange_based_on_PS(MPI_Comm Communicator)
{
  int CommThisTask, CommNTask, CommPTask;
  MPI_Comm_size(Communicator, &CommNTask);
  MPI_Comm_rank(Communicator, &CommThisTask);

  for(CommPTask = 0; CommNTask > (1 << CommPTask); CommPTask++)
    ;

  int *Send_count  = (int *)Mem.mymalloc_movable(&Send_count,  "Send_count",  sizeof(int) * CommNTask);
  int *Send_offset = (int *)Mem.mymalloc_movable(&Send_offset, "Send_offset", sizeof(int) * CommNTask);
  int *Recv_count  = (int *)Mem.mymalloc_movable(&Recv_count,  "Recv_count",  sizeof(int) * CommNTask);
  int *Recv_offset = (int *)Mem.mymalloc_movable(&Recv_offset, "Recv_offset", sizeof(int) * CommNTask);

  int nimport = 0, nexport = 0, nstay = 0, nlocal = 0;

  /* for type_select == 0, we process gas particles; otherwise all other particles */
  for(int type_select = 0; type_select < 2; type_select++)
    {
      unsigned char *Ptype = (unsigned char *)Mem.mymalloc_movable(&Ptype, "Ptype", sizeof(unsigned char) * Tp->NumPart);
      int *Ptask           = (int *)Mem.mymalloc_movable(&Ptask, "Ptask", sizeof(int) * Tp->NumPart);

      for(int i = 0; i < Tp->NumPart; i++)
        {
          Ptype[i] = Tp->P[i].getType();
          Ptask[i] = Tp->PS[i].TargetTask;

          if(Ptype[i] == 0 && i >= Tp->NumGas)
            Terminate("Bummer1");

          if(Ptype[i] != 0 && i < Tp->NumGas)
            Terminate("Bummer2");
        }

      int NumPart_saved = Tp->NumPart;

      /* ------------------ GAS: exchange SphP upfront ------------------ */
      if(type_select == 0)
        {
          sph_particle_data *sphBuf = NULL;

          for(int rep = 0; rep < 2; rep++)
            {
              for(int n = 0; n < CommNTask; n++)
                Send_count[n] = 0;

              nstay = 0;

              for(int n = 0; n < Tp->NumGas; n++)
                {
                  int target = Ptask[n];

                  if(rep == 0)
                    {
                      if(target != CommThisTask)
                        Send_count[target]++;
                      else
                        nstay++;
                    }
                  else
                    {
                      if(target != CommThisTask)
                        sphBuf[Send_offset[target] + Send_count[target]++] = Tp->SphP[n];
                      else
                        Tp->SphP[nstay++] = Tp->SphP[n];
                    }
                }

              if(rep == 0)
                {
                  myMPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, Communicator);

                  nimport = 0; nexport = 0;
                  Recv_offset[0] = Send_offset[0] = 0;

                  for(int j = 0; j < CommNTask; j++)
                    {
                      nexport += Send_count[j];
                      nimport += Recv_count[j];

                      if(j > 0)
                        {
                          Send_offset[j] = Send_offset[j - 1] + Send_count[j - 1];
                          Recv_offset[j] = Recv_offset[j - 1] + Recv_count[j - 1];
                        }
                    }

                  sphBuf = (sph_particle_data *)Mem.mymalloc_movable(&sphBuf, "sphBuf", nexport * sizeof(sph_particle_data));
                }
              else
                {
                  Tp->NumGas += (nimport - nexport);

                  int max_loadsph = Tp->NumGas;
                  MPI_Allreduce(MPI_IN_PLACE, &max_loadsph, 1, MPI_INT, MPI_MAX, Communicator);

                  if(max_loadsph > (1.0 - ALLOC_TOLERANCE) * Tp->MaxPartSph ||
                     max_loadsph < (1.0 - 3 * ALLOC_TOLERANCE) * Tp->MaxPartSph)
                    Tp->reallocate_memory_maxpartsph(max_loadsph / (1.0 - 2 * ALLOC_TOLERANCE));

                  for(int ngrp = 1; ngrp < (1 << CommPTask); ngrp++)
                    {
                      int target = CommThisTask ^ ngrp;

                      if(target < CommNTask)
                        {
                          if(Send_count[target] > 0 || Recv_count[target] > 0)
                            {
                              myMPI_Sendrecv(sphBuf + Send_offset[target], Send_count[target] * sizeof(sph_particle_data), MPI_BYTE,
                                             target, TAG_SPHDATA,
                                             Tp->SphP + Recv_offset[target] + nstay,
                                             Recv_count[target] * sizeof(sph_particle_data), MPI_BYTE,
                                             target, TAG_SPHDATA,
                                             Communicator, MPI_STATUS_IGNORE);
                            }
                        }
                    }

                  Mem.myfree(sphBuf);  // LIFO within this rep-loop is fine (only buffer alive here)
                }
            }
        }

      /* ------------------ Exchange P[] (and DustP[] in parallel) ------------------ */

      pdata *partBuf = NULL;

#ifdef DUST
      dust_data *dustBuf = NULL;
#endif

      for(int rep = 0; rep < 2; rep++)
        {
          for(int n = 0; n < CommNTask; n++)
            Send_count[n] = 0;

          nstay  = 0;
          nlocal = 0;

          for(int n = 0; n < NumPart_saved; n++)
            {
              const bool selected = (Ptype[n] == type_select) || (type_select != 0);

              if(selected)
                {
                  int target = Ptask[n];

                  if(rep == 0)
                    {
                      if(target != CommThisTask)
                        Send_count[target]++;
                      else
                        {
                          nstay++;
                          nlocal++;
                        }
                    }
                  else
                    {
                      if(target != CommThisTask)
                        {
                          const int off = Send_offset[target] + Send_count[target]++;
                          partBuf[off] = Tp->P[n];

#ifdef DUST
                          if(Tp->P[n].getType() == 6)
                            dustBuf[off] = Tp->DustP[n];
                          else
                            memset(&dustBuf[off], 0, sizeof(dust_data));
#endif
                        }
                      else
                        {
                          Tp->P[nstay] = Tp->P[n];

#ifdef DUST
                          if(Tp->P[n].getType() == 6)
                            Tp->DustP[nstay] = Tp->DustP[n];
                          else
                            memset(&Tp->DustP[nstay], 0, sizeof(dust_data));
#endif

                          nstay++;
                          nlocal++;
                        }
                    }
                }
              else
                {
                  // only relevant for type_select == 0 (we keep non-selected particles in place)
                  if(rep == 0)
                    nstay++;
                  else
                    {
                      Tp->P[nstay] = Tp->P[n];

#ifdef DUST
                      if(Tp->P[n].getType() == 6)
                        Tp->DustP[nstay] = Tp->DustP[n];
                      else
                        memset(&Tp->DustP[nstay], 0, sizeof(dust_data));
#endif

                      nstay++;
                    }
                }
            }

          if(rep == 0)
            {
              myMPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, Communicator);

              nimport = 0; nexport = 0;
              Recv_offset[0] = Send_offset[0] = 0;

              for(int j = 0; j < CommNTask; j++)
                {
                  nexport += Send_count[j];
                  nimport += Recv_count[j];

                  if(j > 0)
                    {
                      Send_offset[j] = Send_offset[j - 1] + Send_count[j - 1];
                      Recv_offset[j] = Recv_offset[j - 1] + Recv_count[j - 1];
                    }
                }

              partBuf = (pdata *)Mem.mymalloc_movable(&partBuf, "partBuf", nexport * sizeof(pdata));

#ifdef DUST
              // IMPORTANT: allocate AFTER partBuf, and free BEFORE partBuf (LIFO)
              dustBuf = (dust_data *)Mem.mymalloc_movable(&dustBuf, "dustBuf", nexport * sizeof(dust_data));
#endif
            }
          else
            {
              Tp->NumPart += (nimport - nexport);

              int max_load = Tp->NumPart;
              MPI_Allreduce(MPI_IN_PLACE, &max_load, 1, MPI_INT, MPI_MAX, Communicator);

              if(max_load > (1.0 - ALLOC_TOLERANCE) * Tp->MaxPart || max_load < (1.0 - 3 * ALLOC_TOLERANCE) * Tp->MaxPart)
                Tp->reallocate_memory_maxpart(max_load / (1.0 - 2 * ALLOC_TOLERANCE));

              if(type_select == 0)
                {
                  // create a gap for incoming particles after local gas
                  memmove(static_cast<void *>(Tp->P + nlocal + nimport),
                          static_cast<void *>(Tp->P + nlocal),
                          (nstay - nlocal) * sizeof(pdata));

#ifdef DUST
                  memmove(static_cast<void *>(Tp->DustP + nlocal + nimport),
                          static_cast<void *>(Tp->DustP + nlocal),
                          (nstay - nlocal) * sizeof(dust_data));
#endif
                }

              for(int ngrp = 1; ngrp < (1 << CommPTask); ngrp++)
                {
                  int target = CommThisTask ^ ngrp;

                  if(target < CommNTask)
                    {
                      if(Send_count[target] > 0 || Recv_count[target] > 0)
                        {
                          myMPI_Sendrecv(partBuf + Send_offset[target], Send_count[target] * sizeof(pdata), MPI_BYTE,
                                         target, TAG_PDATA,
                                         Tp->P + Recv_offset[target] + nlocal, Recv_count[target] * sizeof(pdata), MPI_BYTE,
                                         target, TAG_PDATA,
                                         Communicator, MPI_STATUS_IGNORE);

#ifdef DUST
                          myMPI_Sendrecv(dustBuf + Send_offset[target], Send_count[target] * sizeof(dust_data), MPI_BYTE,
                                         target, TAG_DUSTDATA,
                                         Tp->DustP + Recv_offset[target] + nlocal, Recv_count[target] * sizeof(dust_data), MPI_BYTE,
                                         target, TAG_DUSTDATA,
                                         Communicator, MPI_STATUS_IGNORE);
#endif
                        }
                    }
                }

#ifdef DUST
// CRITICAL: Zero DustP for ALL gas particles first
for(int i = 0; i < Tp->NumGas; i++)
  memset(&Tp->DustP[i], 0, sizeof(dust_data));

// sanitize received slots: DustP meaningful only for type 6
for(int i = nlocal; i < nlocal + nimport; i++)
  {
    if(Tp->P[i].getType() != 6)
      memset(&Tp->DustP[i], 0, sizeof(dust_data));
  }
#endif

#ifdef DUST
              Mem.myfree(dustBuf);   // MUST free dustBuf first (last allocated)
#endif
              Mem.myfree(partBuf);   // then partBuf
            }
        }

      /* ------------------ Exchange PS[] (subfind data) ------------------ */

      subfind_data *subBuf = NULL;

      for(int rep = 0; rep < 2; rep++)
        {
          for(int n = 0; n < CommNTask; n++)
            Send_count[n] = 0;

          nstay  = 0;
          nlocal = 0;

          for(int n = 0; n < NumPart_saved; n++)
            {
              const bool selected = (Ptype[n] == type_select) || (type_select != 0);

              if(selected)
                {
                  int target = Ptask[n];

                  if(rep == 0)
                    {
                      if(target != CommThisTask)
                        Send_count[target]++;
                      else
                        {
                          nstay++;
                          nlocal++;
                        }
                    }
                  else
                    {
                      if(target != CommThisTask)
                        subBuf[Send_offset[target] + Send_count[target]++] = Tp->PS[n];
                      else
                        {
                          Tp->PS[nstay++] = Tp->PS[n];
                          nlocal++;
                        }
                    }
                }
              else
                {
                  if(rep == 0)
                    nstay++;
                  else
                    Tp->PS[nstay++] = Tp->PS[n];
                }
            }

          if(rep == 0)
            {
              myMPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, Communicator);

              nimport = 0; nexport = 0;
              Recv_offset[0] = Send_offset[0] = 0;

              for(int j = 0; j < CommNTask; j++)
                {
                  nexport += Send_count[j];
                  nimport += Recv_count[j];

                  if(j > 0)
                    {
                      Send_offset[j] = Send_offset[j - 1] + Send_count[j - 1];
                      Recv_offset[j] = Recv_offset[j - 1] + Recv_count[j - 1];
                    }
                }

              subBuf = (subfind_data *)Mem.mymalloc_movable(&subBuf, "subBuf", nexport * sizeof(subfind_data));
            }
          else
            {
              Tp->PS = (subfind_data *)Mem.myrealloc_movable(Tp->PS, Tp->NumPart * sizeof(subfind_data));

              if(type_select == 0)
                {
                  memmove(Tp->PS + nlocal + nimport,
                          Tp->PS + nlocal,
                          (nstay - nlocal) * sizeof(subfind_data));
                }

              for(int ngrp = 1; ngrp < (1 << CommPTask); ngrp++)
                {
                  int target = CommThisTask ^ ngrp;

                  if(target < CommNTask)
                    {
                      if(Send_count[target] > 0 || Recv_count[target] > 0)
                        {
                          myMPI_Sendrecv(subBuf + Send_offset[target], Send_count[target] * sizeof(subfind_data), MPI_BYTE,
                                         target, TAG_KEY,
                                         Tp->PS + Recv_offset[target] + nlocal, Recv_count[target] * sizeof(subfind_data), MPI_BYTE,
                                         target, TAG_KEY,
                                         Communicator, MPI_STATUS_IGNORE);
                        }
                    }
                }

              Mem.myfree(subBuf);
            }
        }

      // FREE in reverse order of allocation inside type_select loop:
      Mem.myfree(Ptask);
      Mem.myfree(Ptype);
    }

  // top-level frees (reverse order of allocation):
  Mem.myfree(Recv_offset);
  Mem.myfree(Recv_count);
  Mem.myfree(Send_offset);
  Mem.myfree(Send_count);

  /* finally, address desired local order according to PS[].TargetIndex */

  if(Tp->NumGas)
    {
      local_sort_data *mp = (local_sort_data *)Mem.mymalloc("mp", sizeof(local_sort_data) * Tp->NumGas);
      int *Id             = (int *)Mem.mymalloc("Id", sizeof(int) * Tp->NumGas);

      for(int i = 0; i < Tp->NumGas; i++)
        {
          mp[i].index       = i;
          mp[i].targetindex = Tp->PS[i].TargetIndex;
        }

      mycxxsort(mp, mp + Tp->NumGas, compare_local_sort_data_targetindex);

      for(int i = 0; i < Tp->NumGas; i++)
        Id[mp[i].index] = i;

      reorder_gas(Id);

      for(int i = 0; i < Tp->NumGas; i++)
        Id[mp[i].index] = i;

      reorder_PS(Id, 0, Tp->NumGas);

      Mem.myfree(Id);
      Mem.myfree(mp);
    }

  if(Tp->NumPart - Tp->NumGas > 0)
    {
      reorder_P_PS(Tp->NumGas, Tp->NumPart);
    }
}



#include "../data/simparticles.h"
template class domain<simparticles>;

#ifdef LIGHTCONE_PARTICLES
#include "../data/lcparticles.h"
template class domain<lcparticles>;
#endif