 *  At 2048³, the dust population grows to ~15M superparticles by z~3. To
 *  avoid these dominating the gravity tree, newly spawned dust is assigned a gravity timebin of
    max(DUST_MIN_TIMEBIN, HighestActiveTimeBin), ensuring it lands
    on a bin that is synchronized with the current hierarchy.
    At late times HighestActiveTimeBin ≤ 15 so DUST_MIN_TIMEBIN
    dominates; at early times (z > 10) HighestActiveTimeBin can be
    21+, and without the clamp dust would spawn on an unsynchronized
    bin causing a collective gravity hang on multi-node runs.
 *
 *  IMPORTANT: spawn_dust_particle caps the timebin at birth, but Gadget's
 *  ongoing timestep criterion (timestep.cc: get_timestep_grav) can migrate
 *  dust back to short bins after the first gravity force evaluation.
 *  get_timestep_grav() in timestep.cc returns TIMEBASE-1 for all dust particles,
 *  causing timebins_get_bin_and_do_validity_checks to assign the highest
 *  currently-synchronized bin. Dust migrates upward naturally from DUST_MIN_TIMEBIN
 *  within a few sync-points after spawning.
 *
 *  Dust physics (drag, growth, sputtering, etc.) runs every 10 gravity steps
 *  via the cadence guard in update_dust_dynamics(), with dt scaled by 10
 *  to compensate. All routines that receive this scaled dt use the exact
 *  analytical form (1 − exp(−dt/τ)) rather than linear approximations,
 *  so the result is independent of step size as long as τ is well resolved.
