CosmicGrain IC generation suite

Place contents under:
  ~/gadget4/ICs/

Layout:
  ICs/
    add_dust_type.py
    run_music2_suite.sh
    MUSIC2_suite_manifest.csv
    MUSIC2_params/
      48 MUSIC2 config files
    halo295/
    halo308/
    ...
    halo9235/

The halo directories are created automatically by run_music2_suite.sh.

Canonical IC naming:
  IC_halo<HALO>_zoom_<RES>.hdf5

Examples:
  IC_halo3886_zoom_512.hdf5
  IC_halo3886_zoom_1024.hdf5
  IC_halo3886_zoom_2048.hdf5
  IC_halo3886_zoom_4096.hdf5

All configurations:
  levelmin=7
  levelmin_TF=9
  levelmax=9/10/11/12 for 512/1024/2048/4096

Parent seed:
  seed[9]=42424242

Higher-level seeds are nested per halo. halo3886 retains:
  seed[10]=38861010

After each MUSIC2 run, add_dust_type.py modifies the same HDF5 file
in place and verifies NTYPES=7 compatibility with empty PartType6.
