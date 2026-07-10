#!/bin/bash

# Dust diagnostics and details
# Usage: ./print_final_dust_overview.sh <output_file>
# Example: ./print_final_dust_overview.sh output_7.log

if [ $# -eq 0 ]; then
    echo "Usage: $0 <output_file>"
    echo "Example: $0 output_7.log"
    exit 1
fi

OUTPUT_FILE=$1

if [ ! -f "$OUTPUT_FILE" ]; then
    echo "Error: File '$OUTPUT_FILE' not found!"
    exit 1
fi

# Color codes
GREEN='\033[0;32m'
PINK='\033[0;35m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
BLUE='\033[0;34m'
NC='\033[0m'

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║           DUST EVOLUTION DIAGNOSTICS v3                        ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo -e "Output file: ${YELLOW}$OUTPUT_FILE${NC}"
echo ""

# ===== FINAL STATE =====
echo -e "${PINK}━━━━━ FINAL STATE (latest STATISTICS block) ━━━━━${NC}"
# Output format: [DUST|T=0|a=X z=Y] STATISTICS ...
grep "STATISTICS" "$OUTPUT_FILE" | grep -v "HK11" | tail -30
echo ""

# Extract key values -- destruction counts (Thermal/Shock/Astration) live in
# the DESTRUCTION AUDIT block, NOT under "STATISTICS Destroyed by ...:" --
# that text does not appear anywhere in the current log format.
AVG_SIZE=$(grep "STATISTICS Avg grain size:" "$OUTPUT_FILE" | tail -1 | grep -oP 'size:\s+\K[0-9.]+')
FINAL_PARTICLES=$(grep "STATISTICS Particles:" "$OUTPUT_FILE" | tail -1 | grep -oP 'Particles:\s+\K\d+')
THERMAL_COUNT=$(grep "Thermal sputtering:" "$OUTPUT_FILE" | tail -1 | grep -oP 'Thermal sputtering:\s+\K\d+')
SHOCK_COUNT=$(grep "Shock destruction:" "$OUTPUT_FILE" | tail -1 | grep -oP 'Shock destruction:\s+\K\d+')
GROWTH_COUNT=$(grep "STATISTICS Growth events:" "$OUTPUT_FILE" | tail -1 | grep -oP 'events:\s+\K\d+')
EROSION_COUNT=$(grep "STATISTICS Partial erosion events:" "$OUTPUT_FILE" | tail -1 | grep -oP 'events:\s+\K\d+')
COAG_COUNT=$(grep "STATISTICS Coagulation events:" "$OUTPUT_FILE" | tail -1 | grep -oP 'events:\s+\K\d+')
ASTRATION_COUNT=$(grep "Astration:" "$OUTPUT_FILE" | tail -1 | grep -oP 'Astration:\s+\K\d+')

# ===== HASH PERFORMANCE =====
echo -e "${PINK}━━━━━ SPATIAL HASH PERFORMANCE ━━━━━${NC}"
HASH_SEARCHES=$(grep "STATISTICS Hash searches:" "$OUTPUT_FILE" | tail -1 | grep -oP 'searches:\s+\K\d+')
HASH_RATE=$(grep "STATISTICS Hash success rate:" "$OUTPUT_FILE" | tail -1 | grep -oP 'rate:\s+\K[0-9.]+')
HASH_FAILED=$(grep "STATISTICS.*Failed searches:" "$OUTPUT_FILE" | tail -1 | grep -oP 'searches:\s+\K\d+')

if [ -n "$HASH_SEARCHES" ]; then
    echo "  Hash searches: $HASH_SEARCHES"
    if [ -n "$HASH_RATE" ]; then
        echo "  Success rate: ${HASH_RATE}%"
        if (( $(echo "$HASH_RATE < 90" | bc -l) )); then
            echo -e "  ${RED}⚠ Low hash success rate! Hash may need rebuild or radius is too small${NC}"
        else
            echo -e "  ${GREEN}✓ Good hash performance${NC}"
        fi
    fi
    if [ -n "$HASH_FAILED" ] && [ "$HASH_FAILED" -gt 0 ]; then
        echo -e "  ${YELLOW}⚠ $HASH_FAILED failed searches (isolated dust in low-density regions)${NC}"
    fi
else
    echo -e "${YELLOW}No hash statistics found${NC}"
fi
echo ""

# ===== GROWTH SUMMARY =====
echo -e "${PINK}━━━━━ GROWTH SUMMARY ━━━━━${NC}"
GROWTH_LINE=$(grep "STATISTICS Growth events:" "$OUTPUT_FILE" | tail -1)
echo "$GROWTH_LINE"
GROWTH_MASS=$(echo "$GROWTH_LINE" | grep -oP '\d+\.\d+e[+-]\d+' | head -1)

# HK11 diagnostics
echo ""
HK11_BLOCK=$(grep "HK11 GROWTH DIAGNOSTICS" "$OUTPUT_FILE" -A 25 | tail -27)
if [ -n "$HK11_BLOCK" ]; then
    echo "HK11 diagnostic summary (Rank 0):"
    echo "$HK11_BLOCK" | grep -E "(PASSED|Failed|Species|Total mass|f_mol)"

    # f_mol distribution is a single combined line:
    #   "  f_mol: diffuse=%d  moderate=%d  dense=%d  sf=%d"
    # (lowercase field names -- not "Diffuse"/"fmol_diffuse", which never
    # appear in the actual log and never matched anything here before)
    FMOL_LINE=$(echo "$HK11_BLOCK" | grep "f_mol:" | tail -1)
    if [ -n "$FMOL_LINE" ]; then
        echo ""
        echo -e "${CYAN}f_mol distribution (where growth occurs):${NC}"
        echo "  $FMOL_LINE"
        echo -e "  ${CYAN}diffuse(0.05)=atomic HI, moderate(0.2)=mixed, dense(0.5)=H2 clouds, sf(0.8)=star-forming${NC}"
    fi
fi

if [ -n "$GROWTH_COUNT" ] && [ "$GROWTH_COUNT" -gt 1000 ]; then
    echo -e "${GREEN}✓ Growth working well ($GROWTH_COUNT events)${NC}"
elif [ -n "$GROWTH_COUNT" ] && [ "$GROWTH_COUNT" -gt 100 ]; then
    echo -e "${YELLOW}? Growth moderate ($GROWTH_COUNT events)${NC}"
elif [ -n "$GROWTH_COUNT" ] && [ "$GROWTH_COUNT" -gt 0 ]; then
    echo -e "${RED}✗ Growth weak ($GROWTH_COUNT events) — check DustGrowthCalibration${NC}"
else
    echo -e "${RED}✗ No growth detected!${NC}"
fi
echo ""

# ===== COAGULATION SUMMARY =====
echo -e "${PINK}━━━━━ COAGULATION SUMMARY ━━━━━${NC}"
COAG_LINE=$(grep "STATISTICS Coagulation events:" "$OUTPUT_FILE" | tail -1)
if [ -n "$COAG_LINE" ]; then
    echo "$COAG_LINE"
    if [ -n "$COAG_COUNT" ] && [ "$COAG_COUNT" -gt 0 ] && [ -n "$GROWTH_COUNT" ] && [ "$GROWTH_COUNT" -gt 0 ]; then
        COAG_RATIO=$(awk "BEGIN {printf \"%.2f\", $COAG_COUNT/$GROWTH_COUNT}")
        echo -e "  Coagulation/accretion ratio: ${COAG_RATIO}"
        if (( $(echo "$COAG_RATIO > 0.5" | bc -l) )); then
            echo -e "  ${GREEN}✓ Coagulation contributing significantly to grain growth${NC}"
        elif (( $(echo "$COAG_RATIO > 0.1" | bc -l) )); then
            echo -e "  ${YELLOW}? Coagulation minor relative to accretion${NC}"
        else
            echo -e "  ${YELLOW}? Very little coagulation — only dense cold gas qualifies (T<100K, n>DustCoagDensThresh)${NC}"
        fi
    fi
else
    echo -e "${YELLOW}No coagulation statistics found${NC}"
fi
echo ""

# ===== DESTRUCTION SUMMARY =====
# All of this now comes from the DESTRUCTION AUDIT block
# ("=== DESTRUCTION AUDIT (global) ===" ... "==...=="), printed by the same
# print_dust_statistics() call as the STATISTICS block above but under
# different field names -- "Thermal sputtering:"/"Shock destruction:", each
# followed by their own (unprefixed) "full destructions:"/"partial erosion:"
# lines, rather than a single "STATISTICS Total mass eroded" line, which
# does not exist in the current format.
echo -e "${PINK}━━━━━ DESTRUCTION SUMMARY ━━━━━${NC}"
grep "Thermal sputtering:" "$OUTPUT_FILE" | tail -1
grep "Shock destruction:" "$OUTPUT_FILE" | tail -1
THERMAL_PARTIAL_MSUN=$(grep -A2 "Thermal sputtering:" "$OUTPUT_FILE" | tail -3 | grep "partial erosion:" | tail -1 | grep -oP 'erosion:\s+\K[0-9.e+-]+')
SHOCK_PARTIAL_MSUN=$(grep -A2 "Shock destruction:" "$OUTPUT_FILE" | tail -3 | grep "partial erosion:" | tail -1 | grep -oP 'erosion:\s+\K[0-9.e+-]+')
if [ -n "$THERMAL_PARTIAL_MSUN" ]; then
    echo "  Thermal partial erosion: $THERMAL_PARTIAL_MSUN Msun"
fi
if [ -n "$SHOCK_PARTIAL_MSUN" ]; then
    echo "  Shock partial erosion:   $SHOCK_PARTIAL_MSUN Msun"
fi

if [ -n "$THERMAL_COUNT" ] && [ -n "$SHOCK_COUNT" ] && [ "$THERMAL_COUNT" -gt 0 ]; then
    RATIO=$(awk "BEGIN {printf \"%.2f\", $SHOCK_COUNT/$THERMAL_COUNT}")
    echo ""
    if (( $(echo "$RATIO > 2.0" | bc -l) )); then
        echo -e "  Shock:Thermal ratio: ${RED}$RATIO:1 (shocks too aggressive)${NC}"
    elif (( $(echo "$RATIO < 0.01" | bc -l) )); then
        echo -e "  Shock:Thermal ratio: ${YELLOW}$RATIO:1 (thermal dominant — expected if lots of hot gas)${NC}"
    else
        echo -e "  Shock:Thermal ratio: ${GREEN}$RATIO:1 (realistic balance)${NC}"
    fi
fi
echo ""

# ===== ASTRATION SUMMARY =====
echo -e "${PINK}━━━━━ ASTRATION SUMMARY ━━━━━${NC}"
ASTRATION_LINE=$(grep "Astration:" "$OUTPUT_FILE" | tail -1)
if [ -n "$ASTRATION_LINE" ]; then
    echo "$ASTRATION_LINE"
    if [ -n "$ASTRATION_COUNT" ] && [ "$ASTRATION_COUNT" -gt 0 ]; then
        echo -e "  ${CYAN}Astration: dust locked into stars during SF — physical loss channel${NC}"
    fi
    echo -e "  ${YELLOW}Note: this DESTRUCTION AUDIT total is the authoritative, complete count."
    echo -e "  The separate sparse [ASTRATION]/[ASTRATION_CHECK] diagnostic samples"
    echo -e "  elsewhere in the log undercount if summed directly -- don't use those.${NC}"
else
    echo -e "${YELLOW}No astration statistics found${NC}"
fi
echo ""

# ===== DUST TEMPERATURE DISTRIBUTION =====
echo -e "${PINK}━━━━━ DUST TEMPERATURE DISTRIBUTION (latest) ━━━━━${NC}"
TEMP_BLOCK=$(grep "STATISTICS.*K (" "$OUTPUT_FILE" | tail -6)
if [ -n "$TEMP_BLOCK" ]; then
    echo "$TEMP_BLOCK"
    # Check if near-sublimation fraction is high
    NEAR_SUBLIM=$(echo "$TEMP_BLOCK" | grep "Near sublim" | grep -oP '\d+(?=\s+\()' | head -1)
    TOTAL_TEMP=$(grep "STATISTICS Particles:" "$OUTPUT_FILE" | tail -1 | grep -oP '\d+' | head -1)
    if [ -n "$NEAR_SUBLIM" ] && [ -n "$TOTAL_TEMP" ] && [ "$TOTAL_TEMP" -gt 0 ]; then
        HOT_FRAC=$(awk "BEGIN {printf \"%.1f\", 100.0*$NEAR_SUBLIM/$TOTAL_TEMP}")
        echo ""
        if (( $(echo "$HOT_FRAC > 20" | bc -l) )); then
            echo -e "  ${RED}⚠ ${HOT_FRAC}% of grains near sublimation (1000-2000 K) — check DustThermalSputteringTemp${NC}"
        else
            echo -e "  ${GREEN}✓ Near-sublimation fraction: ${HOT_FRAC}%${NC}"
        fi
    fi
else
    echo -e "${YELLOW}No temperature distribution found (look for STATISTICS lines with 'K (')${NC}"
fi
echo ""

# ===== GAS BUDGET =====
echo -e "${PINK}━━━━━ GAS PHASE BUDGET (latest) ━━━━━${NC}"
GAS_BUDGET=$(grep "GAS_BUDGET" "$OUTPUT_FILE" | tail -1)
if [ -n "$GAS_BUDGET" ]; then
    echo "$GAS_BUDGET"
    # Warn if vhot fraction is high (correlates with thermal sputtering)
    VHOT_PCT=$(echo "$GAS_BUDGET" | grep -oP 'vhot=\K[0-9.]+')
    if [ -n "$VHOT_PCT" ]; then
        if (( $(echo "$VHOT_PCT > 50" | bc -l) )); then
            echo -e "  ${RED}⚠ ${VHOT_PCT}% of gas is very hot (T>10^6 K) — expect high thermal sputtering${NC}"
        elif (( $(echo "$VHOT_PCT > 20" | bc -l) )); then
            echo -e "  ${YELLOW}? ${VHOT_PCT}% of gas is very hot — moderate sputtering expected${NC}"
        else
            echo -e "  ${GREEN}✓ vhot gas fraction reasonable (${VHOT_PCT}%)${NC}"
        fi
    fi
else
    echo -e "${YELLOW}No gas budget found (only printed every 500 steps)${NC}"
fi
echo ""

# ===== GRAIN SIZE DISTRIBUTIONS AT KEY REDSHIFTS =====
echo -e "${PINK}━━━━━ GRAIN SIZE EVOLUTION (z=2.0 → 0) ━━━━━${NC}"

TARGET_Z=(2.0 1.5 1.0 0.5 0.0)

for target_z in "${TARGET_Z[@]}"; do
    CLOSEST_LINE=$(grep -n "=== GRAIN SIZE DISTRIBUTION ===" "$OUTPUT_FILE" | while IFS=: read -r linenum _; do
        # FIX: the [DUST|...] prefix (and its z=...) is on the SAME line as
        # the "=== GRAIN SIZE DISTRIBUTION ===" header itself, not the line
        # before it -- every DUST_PRINT call gets its own prefix.
        Z=$(sed -n "${linenum}p" "$OUTPUT_FILE" | grep -oP 'z=\K[0-9.]+')
        if [ -n "$Z" ]; then
            DIFF=$(awk "BEGIN {printf \"%.3f\", sqrt(($Z - $target_z)^2)}")
            echo "$DIFF:$linenum:$Z"
        fi
    done | sort -n | head -1)

    if [ -n "$CLOSEST_LINE" ]; then
        LINENUM=$(echo "$CLOSEST_LINE" | cut -d: -f2)
        ACTUAL_Z=$(echo "$CLOSEST_LINE" | cut -d: -f3)
        echo ""
        echo -e "${BLUE}─── z ≈ $target_z (actual: $ACTUAL_Z) ───${NC}"
        sed -n "$((LINENUM)),$((LINENUM+10))p" "$OUTPUT_FILE" | grep -v "^--$"
    fi
done
echo ""

# ===== EVOLUTION TIMELINE =====
echo -e "${PINK}━━━━━ EVOLUTION TIMELINE (z=2.0 → 0) ━━━━━${NC}"
echo "Redshift | Particles | Avg Size | Thermal | Shocks | Coag    | Growth  | Astrat"
echo "---------|-----------|----------|---------|--------|---------|---------|-------"

grep -n "STATISTICS Particles:" "$OUTPUT_FILE" | while IFS=: read -r linenum _; do
    # Widened to +50 lines (was +20): a full STATISTICS+DESTRUCTION AUDIT
    # combined block from a single print_dust_statistics() call runs to
    # ~40+ lines in practice, so +20 was cutting off the DESTRUCTION AUDIT
    # section (Thermal/Shock/Astration) before it even got the chance to
    # collide with the STATISTICS-only filter bug below.
    RAW_BLOCK=$(sed -n "${linenum},$((linenum+50))p" "$OUTPUT_FILE")
    # STATISTICS-prefixed fields (Particles, Avg size, Growth, Coagulation)
    # correctly stay filtered to "STATISTICS" lines, matching the current
    # format. Thermal/Shock/Astration do NOT carry "STATISTICS" in their
    # own text (they're separate DESTRUCTION AUDIT print calls in the same
    # C++ function) -- FIX: search RAW_BLOCK directly for those three,
    # not the STATISTICS-filtered subset, which silently excluded them
    # entirely before.
    BLOCK=$(echo "$RAW_BLOCK" | grep "STATISTICS")

    Z=$(sed -n "$((linenum))p" "$OUTPUT_FILE" | grep -oP '\bz=\K[0-9.]+')

    if [ -n "$Z" ]; then
        IN_RANGE=$(awk "BEGIN {print ($Z >= 0.0 && $Z <= 2.0)}")
        if [ "$IN_RANGE" -eq 1 ]; then
            PARTS=$(echo "$BLOCK" | grep "Particles:" | grep -oP 'Particles:\s+\K\d+' | head -1)
            SIZE=$(echo "$BLOCK" | grep "Avg grain size:" | grep -oP 'size:\s+\K[0-9.]+' | head -1)
            THERM=$(echo "$RAW_BLOCK" | grep "Thermal sputtering:" | grep -oP 'sputtering:\s+\K\d+' | head -1)
            SHOCK=$(echo "$RAW_BLOCK" | grep "Shock destruction:" | grep -oP 'destruction:\s+\K\d+' | head -1)
            COAG=$(echo "$BLOCK" | grep "Coagulation events:" | grep -oP 'events:\s+\K\d+' | head -1)
            GROW=$(echo "$BLOCK" | grep "Growth events:" | grep -oP 'events:\s+\K\d+' | head -1)
            ASTR=$(echo "$RAW_BLOCK" | grep "Astration:" | grep -oP 'Astration:\s+\K\d+' | head -1)

            if [ -n "$Z" ] && [ -n "$PARTS" ]; then
                printf "%06.3f:z=%-6s | %-9s | %6s nm | %-7s | %-6s | %-7s | %-7s | %s\n" \
                    "$Z" "$Z" "${PARTS}" "${SIZE:-0}" "${THERM:-0}" "${SHOCK:-0}" "${COAG:-0}" "${GROW:-0}" "${ASTR:-0}"
            fi
        fi
    fi
done | sort -rn | cut -d: -f2
echo ""

# ===== FEEDBACK & DUST CREATION =====
# FIX: the tag "DUST_STATS" does not exist anywhere in the source -- this
# never matched anything. The real feedback summary line is printed by
# FB_PRINT (prefix "[FEEDBACK|T=...]") as:
#   events: SNII=%lld (%.3e erg)  AGB=%lld (%.3e erg)
# There is no "Total dust: X Msun" field on this line at all -- dust mass
# created is tracked separately (DESTRUCTION AUDIT's "Total created:" line
# below reports PARTICLE COUNTS, not a mass), so that extraction is dropped
# rather than left silently matching nothing.
echo -e "${PINK}━━━━━ FEEDBACK & DUST CREATION ━━━━━${NC}"
FEEDBACK_LINES=$(grep "events: SNII=" "$OUTPUT_FILE" | tail -5)
if [ -n "$FEEDBACK_LINES" ]; then
    echo "$FEEDBACK_LINES"
    echo ""
    LAST_LINE=$(echo "$FEEDBACK_LINES" | tail -1)
    SNII_N=$(echo "$LAST_LINE" | grep -oP 'SNII=\K\d+')
    SNII_E=$(echo "$LAST_LINE" | grep -oP 'SNII=\d+\s+\(\K[0-9.e+-]+')
    AGB_N=$(echo "$LAST_LINE" | grep -oP 'AGB=\K\d+')
    AGB_E=$(echo "$LAST_LINE" | grep -oP 'AGB=\d+\s+\(\K[0-9.e+-]+')
    if [ -n "$SNII_N" ]; then
        echo -e "${BLUE}→${NC} Latest cumulative feedback events: SNII=$SNII_N ($SNII_E erg)  AGB=$AGB_N ($AGB_E erg)"
    fi
    DUST_CREATED_LINE=$(grep "Total created:" "$OUTPUT_FILE" | tail -1)
    if [ -n "$DUST_CREATED_LINE" ]; then
        echo -e "${BLUE}→${NC} $DUST_CREATED_LINE"
    fi
else
    echo -e "${YELLOW}No FEEDBACK event lines found${NC}"
fi
echo ""

# ===== TIMING =====
echo -e "${PINK}━━━━━ PERFORMANCE ━━━━━${NC}"
TIMING_LINE=$(grep "DUST_TIMING" "$OUTPUT_FILE" | tail -1)
if [ -n "$TIMING_LINE" ]; then
    echo "$TIMING_LINE"
else
    echo -e "${YELLOW}No timing data yet (printed every 100 calls)${NC}"
fi
echo ""

# ===== DIAGNOSTIC CHECKS =====
echo -e "${PINK}━━━━━ DIAGNOSTIC CHECKS ━━━━━${NC}"

# Check 1: Grain growth
if [ -n "$AVG_SIZE" ]; then
    if (( $(echo "$AVG_SIZE > 20.0" | bc -l) )); then
        echo -e "${GREEN}✓${NC} Grain growth successful (${AVG_SIZE} nm)"
    elif (( $(echo "$AVG_SIZE > 5.0" | bc -l) )); then
        echo -e "${GREEN}✓${NC} Grain sizes reasonable (${AVG_SIZE} nm)"
    elif (( $(echo "$AVG_SIZE < 2.0" | bc -l) )); then
        echo -e "${RED}✗${NC} Grains too small (${AVG_SIZE} nm) — growth not working?"
    else
        echo -e "${YELLOW}?${NC} Grain size marginal: ${AVG_SIZE} nm"
    fi
else
    echo -e "${RED}✗${NC} Could not determine grain size (check STATISTICS grep pattern)"
fi

# Check 2: Particle survival
if [ -n "$FINAL_PARTICLES" ]; then
    if [ "$FINAL_PARTICLES" -gt 1000 ]; then
        echo -e "${GREEN}✓${NC} Excellent dust survival ($FINAL_PARTICLES particles)"
    elif [ "$FINAL_PARTICLES" -gt 100 ]; then
        echo -e "${GREEN}✓${NC} Good dust survival ($FINAL_PARTICLES particles)"
    elif [ "$FINAL_PARTICLES" -lt 25 ]; then
        echo -e "${RED}✗${NC} Very few dust particles ($FINAL_PARTICLES) — destruction too strong?"
    else
        echo -e "${YELLOW}?${NC} Marginal survival ($FINAL_PARTICLES particles)"
    fi
else
    echo -e "${RED}✗${NC} Could not determine particle count"
fi

# Check 3: Growth vs destruction
if [ -n "$GROWTH_COUNT" ] && [ -n "$THERMAL_COUNT" ] && [ -n "$SHOCK_COUNT" ]; then
    TOTAL_DESTROYED=$((THERMAL_COUNT + SHOCK_COUNT))
    if [ "$TOTAL_DESTROYED" -gt 0 ] && [ "$GROWTH_COUNT" -gt 0 ]; then
        GROWTH_RATIO=$(awk "BEGIN {printf \"%.2f\", $GROWTH_COUNT/$TOTAL_DESTROYED}")
        if (( $(echo "$GROWTH_RATIO > 1.0" | bc -l) )); then
            echo -e "${GREEN}✓${NC} Growth exceeds destruction (ratio: $GROWTH_RATIO)"
        elif (( $(echo "$GROWTH_RATIO > 0.3" | bc -l) )); then
            echo -e "${GREEN}✓${NC} Growth competitive with destruction (ratio: $GROWTH_RATIO)"
        elif (( $(echo "$GROWTH_RATIO < 0.1" | bc -l) )); then
            echo -e "${RED}✗${NC} Growth too weak relative to destruction (ratio: $GROWTH_RATIO)"
        else
            echo -e "${YELLOW}?${NC} Growth marginal (ratio: $GROWTH_RATIO)"
        fi
    fi
fi

# Check 4: Coagulation contributing?
if [ -n "$COAG_COUNT" ] && [ "$COAG_COUNT" -gt 0 ]; then
    echo -e "${GREEN}✓${NC} Coagulation active ($COAG_COUNT events)"
else
    echo -e "${YELLOW}?${NC} No coagulation — check DustCoagulationDensityThresh (needs T<100K AND dense gas)"
fi

# Check 5: Astration
if [ -n "$ASTRATION_COUNT" ] && [ "$ASTRATION_COUNT" -gt 0 ]; then
    echo -e "${GREEN}✓${NC} Astration active ($ASTRATION_COUNT particles consumed by SF)"
fi

echo ""

# ===== RECOMMENDATIONS =====
echo -e "${PINK}━━━━━ RECOMMENDATIONS ━━━━━${NC}"
NEEDS_HELP=0

if [ -n "$AVG_SIZE" ] && (( $(echo "$AVG_SIZE < 5.0" | bc -l) )); then
    echo -e "${CYAN}→${NC} Growth too slow: Decrease DustGrowthCalibration or check f_mol distribution"
    NEEDS_HELP=1
fi

if [ -n "$RATIO" ] && (( $(echo "$RATIO > 2.0" | bc -l) )); then
    echo -e "${CYAN}→${NC} Shocks too dominant: Reduce shock destruction efficiency or shock radius cap"
    NEEDS_HELP=1
fi

if [ -n "$VHOT_PCT" ] && (( $(echo "$VHOT_PCT > 50" | bc -l) )); then
    echo -e "${CYAN}→${NC} Too much hot gas → thermal sputtering dominant: Increase DustThermalSputteringTemp"
    NEEDS_HELP=1
fi

if [ -n "$COAG_COUNT" ] && [ "$COAG_COUNT" -eq 0 ]; then
    echo -e "${CYAN}→${NC} No coagulation: Lower DustCoagulationDensityThresh or check gas is reaching T<100K"
    NEEDS_HELP=1
fi

if [ -n "$FINAL_PARTICLES" ] && [ "$FINAL_PARTICLES" -lt 50 ]; then
    echo -e "${CYAN}→${NC} ${RED}CRITICAL${NC}: Very few surviving dust particles ($FINAL_PARTICLES)"
    NEEDS_HELP=1
fi

if [ "$NEEDS_HELP" -eq 0 ]; then
    echo -e "${GREEN}✓ All systems nominal!${NC}"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    END OF DIAGNOSTICS                          ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
