/*******************************************************************************/
/*
  This performs 2D interpolation that converts MESA stellar evolution models into a fast lookup table, 
  allowing the simulation to quickly ask "what yields does a 2.5 Msun, Z=0.015 AGB star produce?" for example
*/
/*******************************************************************************/
#include "gadgetconfig.h"

#include "agb_yields.h"
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>

// Global instance
AGBYieldTable AGB_Yields;

AGBYieldTable::AGBYieldTable() 
    : is_loaded(false)
{
    reset_diagnostics();
}

bool AGBYieldTable::load_from_file(const char* filename) {
    std::ifstream file(filename);
    if(!file.is_open()) {
        printf("[AGB_YIELDS] ERROR: Cannot open yield table: %s\n", filename);
        return false;
    }
    
    table.clear();
    mass_grid.clear();
    Z_grid.clear();
    
    std::string line;
    int line_count = 0;
    int data_lines = 0;
    
    while(std::getline(file, line)) {
        line_count++;
        
        // Skip comments and empty lines
        if(line.empty() || line[0] == '#') continue;
        
        std::istringstream iss(line);
        AGBYieldEntry entry;
        
        // Parse line: mass Z t_start t_end C N O Ne Mg Z_total M_lost
        if(!(iss >> entry.mass_init >> entry.Z_init >> 
             entry.t_AGB_start >> entry.t_AGB_end >> 
             entry.C_yield >> entry.N_yield >> entry.O_yield >>
             entry.Ne_yield >> entry.Mg_yield >>
             entry.Z_yield_total >> entry.M_lost)) {
            printf("[AGB_YIELDS] WARNING: Skipping malformed line %d\n", line_count);
            continue;
        }
        
        table.push_back(entry);
        data_lines++;
        
        // Build unique grids
        if(std::find(mass_grid.begin(), mass_grid.end(), entry.mass_init) == mass_grid.end())
            mass_grid.push_back(entry.mass_init);
        if(std::find(Z_grid.begin(), Z_grid.end(), entry.Z_init) == Z_grid.end())
            Z_grid.push_back(entry.Z_init);
    }
    
    file.close();
    
    if(data_lines == 0) {
        printf("[AGB_YIELDS] ERROR: No valid data lines found\n");
        return false;
    }
    
    // Sort grids for interpolation
    std::sort(mass_grid.begin(), mass_grid.end());
    std::sort(Z_grid.begin(), Z_grid.end());
    
    is_loaded = true;
    
    printf("[AGB_YIELDS] Successfully loaded yield table:\n");
    printf("[AGB_YIELDS]   File: %s\n", filename);
    printf("[AGB_YIELDS]   Entries: %zu\n", table.size());
    printf("[AGB_YIELDS]   Mass range: %.2f - %.2f Msun (%zu points)\n",
           mass_grid.front(), mass_grid.back(), mass_grid.size());
    printf("[AGB_YIELDS]   Z range: %.6f - %.6f (%zu points)\n",
           Z_grid.front(), Z_grid.back(), Z_grid.size());
    
    // Print first few entries as sanity check
    printf("[AGB_YIELDS]   Sample entries:\n");
    for(size_t i = 0; i < std::min(size_t(3), table.size()); i++) {
        printf("[AGB_YIELDS]     M=%.2f Z=%.4f: Z_yield=%.4f M_lost=%.3f\n",
               table[i].mass_init, table[i].Z_init, 
               table[i].Z_yield_total, table[i].M_lost);
    }
    
    return true;
}

void AGBYieldTable::find_mass_bracket(double mass, int& i_low, int& i_high, double& frac) {
    // Find bracketing indices in mass_grid
    
    // Clamp to table bounds
    if(mass <= mass_grid.front()) {
        i_low = 0;
        i_high = 0;
        frac = 0.0;
        return;
    }
    if(mass >= mass_grid.back()) {
        i_low = mass_grid.size() - 1;
        i_high = mass_grid.size() - 1;
        frac = 0.0;
        return;
    }
    
    // Binary search for bracket
    auto upper = std::lower_bound(mass_grid.begin(), mass_grid.end(), mass);
    i_high = upper - mass_grid.begin();
    i_low = i_high - 1;
    
    // Linear interpolation fraction
    frac = (mass - mass_grid[i_low]) / (mass_grid[i_high] - mass_grid[i_low]);
}

void AGBYieldTable::find_Z_bracket(double Z, int& i_low, int& i_high, double& frac) {
    // Find bracketing indices in Z_grid
    
    // Clamp to table bounds
    if(Z <= Z_grid.front()) {
        i_low = 0;
        i_high = 0;
        frac = 0.0;
        return;
    }
    if(Z >= Z_grid.back()) {
        i_low = Z_grid.size() - 1;
        i_high = Z_grid.size() - 1;
        frac = 0.0;
        return;
    }
    
    // Binary search for bracket
    auto upper = std::lower_bound(Z_grid.begin(), Z_grid.end(), Z);
    i_high = upper - Z_grid.begin();
    i_low = i_high - 1;
    
    // Linear interpolation fraction
    frac = (Z - Z_grid[i_low]) / (Z_grid[i_high] - Z_grid[i_low]);
}

AGBYieldEntry AGBYieldTable::interpolate_yields(double mass, double Z) {
    if(!is_loaded) {
        AGBYieldEntry empty = {};
        return empty;
    }
    
    lookup_count++;
    
    // Track statistics
    if(mass < min_mass_requested) min_mass_requested = mass;
    if(mass > max_mass_requested) max_mass_requested = mass;
    sum_mass += mass;
    
    if(Z < min_Z_requested) min_Z_requested = Z;
    if(Z > max_Z_requested) max_Z_requested = Z;
    sum_Z += Z;
    
    // Track extrapolation
    if(mass < mass_grid.front()) mass_below_table++;
    if(mass > mass_grid.back()) mass_above_table++;
    if(Z < Z_grid.front()) Z_below_table++;
    if(Z > Z_grid.back()) Z_above_table++;
    
    // Find bracketing grid points
    int m_low, m_high, z_low, z_high;
    double m_frac, z_frac;
    
    find_mass_bracket(mass, m_low, m_high, m_frac);
    find_Z_bracket(Z, z_low, z_high, z_frac);
    
    // Find the 4 corners in the table
    AGBYieldEntry corner[2][2];  // [mass_index][Z_index]
    bool found[2][2] = {{false, false}, {false, false}};
    
    for(const auto& entry : table) {
        // Match mass indices
        int m_idx = -1;
        if(fabs(entry.mass_init - mass_grid[m_low]) < 1e-6) m_idx = 0;
        else if(m_low != m_high && fabs(entry.mass_init - mass_grid[m_high]) < 1e-6) m_idx = 1;
        
        if(m_idx < 0) continue;
        
        // Match Z indices
        int z_idx = -1;
        if(fabs(entry.Z_init - Z_grid[z_low]) < 1e-6) z_idx = 0;
        else if(z_low != z_high && fabs(entry.Z_init - Z_grid[z_high]) < 1e-6) z_idx = 1;
        
        if(z_idx < 0) continue;
        
        corner[m_idx][z_idx] = entry;
        found[m_idx][z_idx] = true;
    }
    
    // Check if we found all corners (or at least one)
    if(!found[0][0] && !found[0][1] && !found[1][0] && !found[1][1]) {
        interpolation_warnings++;
        if(interpolation_warnings <= 10) {
            printf("[AGB_YIELDS] WARNING: No matching entries for M=%.2f Z=%.4f\n", mass, Z);
        }
        AGBYieldEntry empty = {};
        return empty;
    }
    
    // Bilinear interpolation
    AGBYieldEntry result = {};
    result.mass_init = mass;
    result.Z_init = Z;
    
    // If we're at a grid point, return that directly
    if(m_frac == 0.0 && z_frac == 0.0 && found[0][0]) {
        return corner[0][0];
    }
    
    // Otherwise do bilinear interpolation
    double w[2][2];  // Weights
    w[0][0] = (1.0 - m_frac) * (1.0 - z_frac);
    w[0][1] = (1.0 - m_frac) * z_frac;
    w[1][0] = m_frac * (1.0 - z_frac);
    w[1][1] = m_frac * z_frac;
    
    // Normalize weights (in case we're missing corners)
    double w_sum = 0.0;
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 2; j++) {
            if(found[i][j]) w_sum += w[i][j];
        }
    }
    
    if(w_sum > 0.0) {
        for(int i = 0; i < 2; i++) {
            for(int j = 0; j < 2; j++) {
                if(found[i][j]) {
                    double weight = w[i][j] / w_sum;
                    result.t_AGB_start += weight * corner[i][j].t_AGB_start;
                    result.t_AGB_end += weight * corner[i][j].t_AGB_end;
                    result.C_yield += weight * corner[i][j].C_yield;
                    result.N_yield += weight * corner[i][j].N_yield;
                    result.O_yield += weight * corner[i][j].O_yield;
                    result.Ne_yield += weight * corner[i][j].Ne_yield;
                    result.Mg_yield += weight * corner[i][j].Mg_yield;
                    result.Z_yield_total += weight * corner[i][j].Z_yield_total;
                    result.M_lost += weight * corner[i][j].M_lost;
                }
            }
        }
    }
    
    return result;
}

double AGBYieldTable::get_total_metal_yield(double mass_msun, double Z_star) {
    if(!is_loaded) return 0.0;
    
    AGBYieldEntry yields = interpolate_yields(mass_msun, Z_star);
    return yields.Z_yield_total;
}

double AGBYieldTable::get_C_yield(double mass_msun, double Z_star) {
    if(!is_loaded) return 0.0;
    AGBYieldEntry yields = interpolate_yields(mass_msun, Z_star);
    return yields.C_yield;
}

double AGBYieldTable::get_N_yield(double mass_msun, double Z_star) {
    if(!is_loaded) return 0.0;
    AGBYieldEntry yields = interpolate_yields(mass_msun, Z_star);
    return yields.N_yield;
}

double AGBYieldTable::get_O_yield(double mass_msun, double Z_star) {
    if(!is_loaded) return 0.0;
    AGBYieldEntry yields = interpolate_yields(mass_msun, Z_star);
    return yields.O_yield;
}

double AGBYieldTable::get_mass_lost(double mass_msun, double Z_star) {
    if(!is_loaded) return 0.0;
    AGBYieldEntry yields = interpolate_yields(mass_msun, Z_star);
    return yields.M_lost;
}

void AGBYieldTable::print_diagnostics() const {
    if(lookup_count == 0) {
        printf("[AGB_YIELDS] No lookups yet\n");
        return;
    }
    
    double avg_mass = sum_mass / lookup_count;
    double avg_Z = sum_Z / lookup_count;
    
    // Single compact line with key stats
    printf("[AGB_YIELDS] Lookups: %d | Warnings: %d | "
           "Mass: %.2f Msun [%.2f-%.2f] | Z: %.4f [%.5f-%.4f]",
           lookup_count, interpolation_warnings,
           avg_mass, min_mass_requested, max_mass_requested,
           avg_Z, min_Z_requested, max_Z_requested);
    
    // Add extrapolation info if relevant
    if(mass_below_table > 0 || mass_above_table > 0 || 
       Z_below_table > 0 || Z_above_table > 0) {
        printf(" | Extrap: M[-%d,+%d] Z[-%d,+%d]",
               mass_below_table, mass_above_table,
               Z_below_table, Z_above_table);
    }
    
    printf("\n");
}

void AGBYieldTable::reset_diagnostics() {
    lookup_count = 0;
    interpolation_warnings = 0;
    
    // Initialize mass statistics
    min_mass_requested = 1e10;
    max_mass_requested = 0.0;
    sum_mass = 0.0;
    
    // Initialize metallicity statistics
    min_Z_requested = 1e10;
    max_Z_requested = 0.0;
    sum_Z = 0.0;
    
    // Initialize extrapolation counters
    mass_below_table = 0;
    mass_above_table = 0;
    Z_below_table = 0;
    Z_above_table = 0;
}