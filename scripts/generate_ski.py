#!/usr/bin/env python3
"""
generate_ski.py
--------------------------------------------------------------------------
Generates a SKIRT .ski file's dust <media> block programmatically from
export_skirt_inputs.py's dust_size_bins*.txt manifest, instead of
hand-transcribing 40+ near-identical ParticleMedium blocks per run.

Why this exists: the dust media section is entirely determined by which
GrainRadius bins actually got populated in a given export -- which depends
on the aperture, which now defaults to R200 and therefore changes per
snapshot/run. Hand-copying blocks from a previous run's .ski file risks
exactly the kind of staleness baked into an earlier file (a header comment
claiming "aperture ~= 66.06 pkpc", which we've since confirmed is wrong --
real R200 for this halo/snapshot is ~177-182 pkpc). This script regenerates
the media block, and the aperture-dependent spatial grid / instrument FOV,
fresh from the actual manifest and actual aperture every time.

Usage:
    # Baseline (SNII+AGB dust, size-binned, no LRN):
    python generate_ski.py skirt_inputs/snap047/ \
        --snap 047 --output halo569_snap047_no_lrn.ski

    # With LRN dust added as its own size-binned set (from dust_size_bins_lrn.txt):
    python generate_ski.py skirt_inputs/snap047/ \
        --snap 047 --lrn binned --output halo569_snap047_with_lrn.ski

Still NOT automated (carried forward as TODOs in the output, same as the
original hand-built file -- this script doesn't invent verification it
can't perform):
  - BpassSEDFamily column order -- verify against your installed SKIRT build
  - stars.txt age column units vs BPASS's expected default
  - numPackets convergence (starts at 1e7 for validation only)
  - instrument distance (10 Mpc placeholder) -- set per real target
--------------------------------------------------------------------------
"""

import argparse
from pathlib import Path


def read_manifest(path):
    """Parse a dust_size_bins*.txt manifest into a list of bin dicts."""
    bins = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            bins.append(dict(
                bin_id=int(parts[0]),
                min_nm=float(parts[1]), max_nm=float(parts[2]), rep_nm=float(parts[3]),
                n=int(parts[4]), m_sil=float(parts[5]), m_car=float(parts[6]),
            ))
    return bins


def read_aperture(export_dir):
    """Pull the aperture (pkpc) this export actually used, from
    export_summary.txt written by process_snapshot()."""
    summary = Path(export_dir) / "export_summary.txt"
    if not summary.exists():
        return None
    for line in summary.read_text().splitlines():
        if line.startswith("Aperture:"):
            return float(line.split(":")[1].strip().split()[0])
    return None


def particle_medium_block(filename, composition_class, rep_nm):
    rep_micron = rep_nm / 1000.0
    return f'''          <ParticleMedium filename="{filename}" massType="Mass" massFraction="1"
                          importMetallicity="false" importTemperature="false"
                          importVelocity="false" importMagneticField="false"
                          importVariableMixParams="false">
            <smoothingKernel type="SmoothingKernel">
              <CubicSplineSmoothingKernel/>
            </smoothingKernel>
            <materialMix type="MaterialMix">
              <ConfigurableDustMix scatteringType="HenyeyGreenstein">
                <populations type="GrainPopulation">
                  <GrainPopulation numSizes="1"
                                   normalizationType="FactorOnSizeDistribution"
                                   factorOnSizeDistribution="1">
                    <composition type="GrainComposition">
                      <{composition_class}/>
                    </composition>
                    <sizeDistribution type="GrainSizeDistribution">
                      <SingleGrainSizeDistribution size="{rep_micron:.9f} micron"/>
                    </sizeDistribution>
                  </GrainPopulation>
                </populations>
              </ConfigurableDustMix>
            </materialMix>
          </ParticleMedium>'''


def build_media_block(bins, label="", path_prefix=""):
    """One silicate + one carbon ParticleMedium per populated bin, skipping
    bins/components with zero mass (mirrors export_size_binned_dust's own
    behavior of not writing empty bin files).

    path_prefix is prepended to every filename so the .ski file references
    the correct location regardless of which directory `skirt` is invoked
    from -- e.g. "skirt_inputs/snap047/" so SKIRT looks there instead of
    wherever the .ski file itself happens to sit."""
    blocks = []
    populated_ids = []
    for b in bins:
        if b["m_sil"] <= 0 and b["m_car"] <= 0:
            continue
        populated_ids.append(b["bin_id"])
        if b["m_sil"] > 0:
            blocks.append(
                f'          <!-- Silicate{label}, size bin {b["bin_id"]:02d}: '
                f'{b["min_nm"]:.3f}-{b["max_nm"]:.3f} nm, rep={b["rep_nm"]:.3f} nm -->'
            )
            blocks.append(particle_medium_block(
                f"{path_prefix}dust_silicate{label}_bin{b['bin_id']:02d}.txt",
                "DraineSilicateGrainComposition", b["rep_nm"]))
        if b["m_car"] > 0:
            blocks.append(
                f'          <!-- Carbon{label}, size bin {b["bin_id"]:02d}: '
                f'{b["min_nm"]:.3f}-{b["max_nm"]:.3f} nm, rep={b["rep_nm"]:.3f} nm -->'
            )
            blocks.append(particle_medium_block(
                f"{path_prefix}dust_carbon{label}_bin{b['bin_id']:02d}.txt",
                "DraineGraphiteGrainComposition", b["rep_nm"]))
    return "\n".join(blocks), populated_ids


SKI_TEMPLATE = '''<?xml version="1.0" encoding="UTF-8"?>
<!-- CosmicGrain -> SKIRT 9 mock spectra, Halo 569, snapshot {snap} -->
<!-- AUTO-GENERATED by generate_ski.py from dust_size_bins.txt{lrn_manifest_note} --
     do not hand-edit the <media> block below; regenerate instead.
     - Aperture used for this export: {aperture_pkpc:.2f} pkpc (from halo_utils R200)
     - SNII+AGB populated bins: {populated_summary}{lrn_populated_line}
     - numPackets: {num_packets}{packets_note}
     - Instrument distance: {distance}{distance_note}
     -->

<!-- STILL MANUAL / NOT VERIFIED -- carried over unchanged from the original
     hand-built file, this script does not check these:
    1. BpassSEDFamily column order: CONFIRMED correct (initialMass, metallicity,
       age matches stars.txt) against BpassSEDFamily::parameterInfo() in the
       SKIRT source. Age units: CONFIRMED the family expects a genuine time
       quantity that SKIRT unit-converts from the file's declared header unit
       (same mechanism as the position-unit fix) -- pending a live column-log
       check of an actual run as final confirmation.
    2. numPackets convergence: NOT tested. See note above.
    3. Instrument distance: deliberate design choice (generic nearby-analog),
       not an unresolved item.
-->
<skirt-simulation-hierarchy type="MonteCarloSimulation" format="9">

  <MonteCarloSimulation userLevel="Expert" simulationMode="DustEmission" numPackets="{num_packets}">

    <units type="Units">
      <ExtragalacticUnits wavelengthOutputStyle="Wavelength" fluxOutputStyle="Frequency"/>
    </units>

    <sourceSystem type="SourceSystem">
      <SourceSystem minWavelength="0.01 micron" maxWavelength="160 micron"
                    wavelengths="0.55 micron" sourceBias="0.5">
        <sources type="Source">
          <ParticleSource filename="{stars_file}" importVelocity="false"
                          importVelocityDispersion="false" importCurrentMass="false"
                          useColumns="">
            <smoothingKernel type="SmoothingKernel">
              <CubicSplineSmoothingKernel/>
            </smoothingKernel>
            <sedFamily type="SEDFamily">
              <BpassSEDFamily imf="Chabrier300" resolution="Downsampled"/>
            </sedFamily>
            <wavelengthBiasDistribution type="WavelengthDistribution">
              <LogWavelengthDistribution minWavelength="0.01 micron" maxWavelength="160 micron"/>
            </wavelengthBiasDistribution>
          </ParticleSource>
        </sources>
      </SourceSystem>
    </sourceSystem>

    <mediumSystem type="MediumSystem">
      <MediumSystem>

        <photonPacketOptions type="PhotonPacketOptions">
          <PhotonPacketOptions minWeightReduction="1e4" minScattEvents="0"
                               pathLengthBias="0.5"/>
        </photonPacketOptions>

        <radiationFieldOptions type="RadiationFieldOptions">
          <RadiationFieldOptions storeRadiationField="true">
            <radiationFieldWLG type="DisjointWavelengthGrid">
              <LogWavelengthGrid minWavelength="0.01 micron" maxWavelength="160 micron" numWavelengths="200"/>
            </radiationFieldWLG>
          </RadiationFieldOptions>
        </radiationFieldOptions>

        <dustEmissionOptions type="DustEmissionOptions">
          <DustEmissionOptions dustEmissionType="Stochastic" sourceWeight="1" wavelengthBias="0.5">
            <dustEmissionWLG type="DisjointWavelengthGrid">
              <NestedLogWavelengthGrid minWavelengthBaseGrid="1 micron" maxWavelengthBaseGrid="1000 micron" numWavelengthsBaseGrid="100" minWavelengthSubGrid="3 micron" maxWavelengthSubGrid="30 micron" numWavelengthsSubGrid="75"/>
            </dustEmissionWLG>
            <wavelengthBiasDistribution type="WavelengthDistribution">
              <LogWavelengthDistribution minWavelength="1 micron" maxWavelength="1000 micron"/>
            </wavelengthBiasDistribution>
          </DustEmissionOptions>
        </dustEmissionOptions>

        <media type="Medium">

{media_block}
{lrn_block}
        </media>

        <grid type="SpatialGrid">
          <PolicyTreeSpatialGrid minX="-{grid_half_pc:.0f} pc" maxX="{grid_half_pc:.0f} pc"
                                 minY="-{grid_half_pc:.0f} pc" maxY="{grid_half_pc:.0f} pc"
                                 minZ="-{grid_half_pc:.0f} pc" maxZ="{grid_half_pc:.0f} pc" treeType="OctTree">
            <policy type="TreePolicy">
              <DensityTreePolicy minLevel="4" maxLevel="9" maxDustFraction="1e-4"
                                 maxDustOpticalDepth="0" wavelength="0.55 micron"
                                 maxDustDensityDispersion="0" maxElectronFraction="1e-6"
                                 maxGasFraction="1e-6"/>
            </policy>
          </PolicyTreeSpatialGrid>
        </grid>

      </MediumSystem>
    </mediumSystem>

    <instrumentSystem type="InstrumentSystem">
      <InstrumentSystem>
        <defaultWavelengthGrid type="WavelengthGrid">
          <LogWavelengthGrid minWavelength="0.1 micron" maxWavelength="1000 micron" numWavelengths="400"/>
        </defaultWavelengthGrid>
        <instruments type="Instrument">

          <FullInstrument instrumentName="faceon" distance="{distance}" inclination="0 deg"
                          azimuth="0 deg" roll="0 deg" fieldOfViewX="{fov_pc:.0f} pc" numPixelsX="500"
                          centerX="0 pc" fieldOfViewY="{fov_pc:.0f} pc" numPixelsY="500" centerY="0 pc"
                          recordComponents="true" numScatteringLevels="0"
                          recordPolarization="false" recordStatistics="false"/>

          <FullInstrument instrumentName="edgeon" distance="{distance}" inclination="90 deg"
                          azimuth="0 deg" roll="0 deg" fieldOfViewX="{fov_pc:.0f} pc" numPixelsX="500"
                          centerX="0 pc" fieldOfViewY="{fov_pc:.0f} pc" numPixelsY="500" centerY="0 pc"
                          recordComponents="true" numScatteringLevels="0"
                          recordPolarization="false" recordStatistics="false"/>

        </instruments>
      </InstrumentSystem>
    </instrumentSystem>

    <probeSystem type="ProbeSystem">
      <ProbeSystem/>
    </probeSystem>

  </MonteCarloSimulation>
</skirt-simulation-hierarchy>
'''


def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("export_dir", help="Directory containing dust_size_bins.txt, "
                                       "export_summary.txt, and the dust_*_binNN.txt files")
    p.add_argument("--snap", default="???", help="Snapshot label for the header comment")
    p.add_argument("--stars", default="stars.txt", help="Stars filename (relative)")
    p.add_argument("--aperture", type=float, default=None,
                   help="Aperture in pkpc, if export_summary.txt isn't available")
    p.add_argument("--grid-margin", type=float, default=1.3,
                   help="Spatial grid half-extent = aperture_pkpc * 1000 * this factor "
                        "(default 1.3; the original hand-built 66pkpc/±80000pc file "
                        "used ~1.21 -- bumped slightly for margin)")
    p.add_argument("--lrn", choices=["binned", "none"], default="binned",
                   help="'binned' (default) includes LRN dust as part of the combined "
                        "SNII+AGB+LRN dust model, read from dust_size_bins_lrn.txt. "
                        "'none' excludes it, if ever needed.")
    p.add_argument("--num-packets", default="1e7",
                   help="numPackets for the simulation. Default 1e7 is a validation-only "
                        "value -- run a convergence sweep (compare total dust luminosity "
                        "across increasing values) before trusting anything above 1e7.")
    p.add_argument("--distance", default="10 Mpc",
                   help="Instrument distance for the mock observation, e.g. '10 Mpc'. "
                        "Default is a generic nearby-analog round number -- this is a "
                        "deliberate design choice (not matching a specific real target), "
                        "confirmed as the intended convention for this pipeline.")
    p.add_argument("--output", required=True, help="Output .ski file path")
    args = p.parse_args()

    export_dir = Path(args.export_dir)
    bins = read_manifest(export_dir / "dust_size_bins.txt")

    aperture_pkpc = args.aperture or read_aperture(export_dir)
    if aperture_pkpc is None:
        raise SystemExit(
            "Could not determine aperture: no export_summary.txt found and "
            "--aperture not given. Refusing to guess a spatial grid size."
        )

    # Prefix every data-file reference with the export directory so the .ski
    # file finds stars.txt/dust_*.txt correctly regardless of which directory
    # `skirt` itself is invoked from (assumes skirt is run from the same cwd
    # as this generator -- true for the scripts/ + skirt_inputs/snapNNN/
    # layout in use here).
    data_dir = str(export_dir).rstrip("/") + "/"
    stars_path = data_dir + args.stars

    media_block, populated_ids = build_media_block(bins, path_prefix=data_dir)
    populated_summary = ", ".join(str(i) for i in populated_ids) if populated_ids else "none"

    lrn_block = ""
    lrn_populated_line = ""
    lrn_manifest_note = ""
    if args.lrn == "binned":
        lrn_manifest_path = export_dir / "dust_size_bins_lrn.txt"
        if not lrn_manifest_path.exists():
            raise SystemExit(
                f"--lrn binned requested but {lrn_manifest_path} doesn't exist -- "
                "rerun export_skirt_inputs.py without --no-lrn first."
            )
        lrn_bins = read_manifest(lrn_manifest_path)
        lrn_block, lrn_populated = build_media_block(lrn_bins, label="_lrn", path_prefix=data_dir)
        lrn_populated_summary = ", ".join(str(i) for i in lrn_populated) if lrn_populated else "none"
        lrn_populated_line = f"\n     - LRN populated bins: {lrn_populated_summary}"
        lrn_manifest_note = " and dust_size_bins_lrn.txt"

    grid_half_pc = aperture_pkpc * 1000.0 * args.grid_margin
    fov_pc = 2.0 * grid_half_pc

    content = SKI_TEMPLATE.format(
        snap=args.snap,
        lrn_manifest_note=lrn_manifest_note,
        aperture_pkpc=aperture_pkpc,
        populated_summary=populated_summary,
        lrn_populated_line=lrn_populated_line,
        num_packets=args.num_packets,
        packets_note=" [DEFAULT -- run a convergence sweep before trusting this]"
                      if args.num_packets == "1e7" else " [explicitly set]",
        distance=args.distance,
        distance_note=" [generic nearby-analog, deliberate — not matching a specific target]",
        stars_file=stars_path,
        media_block=media_block,
        lrn_block=lrn_block,
        grid_half_pc=grid_half_pc,
        fov_pc=fov_pc,
    )

    Path(args.output).write_text(content)
    print(f"Wrote {args.output}")
    print(f"  Aperture: {aperture_pkpc:.2f} pkpc -> grid half-extent {grid_half_pc:.0f} pc, "
          f"instrument FOV {fov_pc:.0f} pc")
    print(f"  numPackets: {args.num_packets}"
          + (" [DEFAULT -- still needs a convergence sweep before a real run]"
             if args.num_packets == "1e7" else " [explicitly set]"))
    print(f"  Distance: {args.distance} [generic nearby-analog, deliberate]")
    print(f"  SNII+AGB: {len(populated_ids)}/{len(bins)} bins populated: {populated_summary}")
    if args.lrn == "binned":
        print(f"  LRN: {len(lrn_populated)}/{len(lrn_bins)} bins populated: {lrn_populated_summary}")


if __name__ == "__main__":
    main()
