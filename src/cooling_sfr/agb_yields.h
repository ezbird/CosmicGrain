#ifndef AGB_YIELDS_H
#define AGB_YIELDS_H

#include "gadgetconfig.h"

#include <vector>

/* Simple AGB yield table for one-time enrichment events
 * Reads MESA-based yield table and provides interpolation
 */

struct AGBYieldEntry {
    double mass_init;        // Initial stellar mass (Msun)
    double Z_init;           // Initial metallicity
    double t_AGB_start;      // AGB start time (Myr)
    double t_AGB_end;        // AGB end time (Myr)
    double C_yield;          // Carbon yield (Msun)
    double N_yield;          // Nitrogen yield (Msun)
    double O_yield;          // Oxygen yield (Msun)
    double Ne_yield;         // Neon yield (Msun)
    double Mg_yield;         // Magnesium yield (Msun)
    double Z_yield_total;    // Total metal yield (Msun)
    double M_lost;           // Total mass lost (Msun)
};

class AGBYieldTable {
private:
    std::vector<AGBYieldEntry> table;
    std::vector<double> mass_grid;      // Unique masses in table
    std::vector<double> Z_grid;         // Unique metallicities in table
    
    bool is_loaded;
    
    // Diagnostics
    int lookup_count;
    int interpolation_warnings;
    
    double min_mass_requested, max_mass_requested, sum_mass;
    double min_Z_requested, max_Z_requested, sum_Z;
    int mass_below_table, mass_above_table;
    int Z_below_table, Z_above_table;

public:
    AGBYieldTable();
    
    // Load table from file
    bool load_from_file(const char* filename);
    
    // Get total metal yield for a star of given mass and Z
    // Returns: Metal yield in Msun (total from entire AGB phase)
    double get_total_metal_yield(double mass_msun, double Z_star);
    
    // Get individual element yields (for future multi-species tracking)
    double get_C_yield(double mass_msun, double Z_star);
    double get_N_yield(double mass_msun, double Z_star);
    double get_O_yield(double mass_msun, double Z_star);
    
    // Get mass lost during AGB
    double get_mass_lost(double mass_msun, double Z_star);
    
    // Check if loaded
    bool is_table_loaded() const { return is_loaded; }
    
    // Diagnostics
    void print_diagnostics() const;
    void reset_diagnostics();
    int get_lookup_count() const { return lookup_count; }
    
private:
    // Bilinear interpolation for given mass and metallicity
    AGBYieldEntry interpolate_yields(double mass, double Z);
    
    // Find bracketing indices
    void find_mass_bracket(double mass, int& i_low, int& i_high, double& frac);
    void find_Z_bracket(double Z, int& i_low, int& i_high, double& frac);
};

// Global instance
extern AGBYieldTable AGB_Yields;

#endif // AGB_YIELDS_H