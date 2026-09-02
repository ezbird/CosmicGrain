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

    double C_yield;          // Total C ejected during AGB (Msun)
    double N_yield;          // Total N ejected during AGB (Msun)
    double O_yield;          // Total O ejected during AGB (Msun)
    double Ne_yield;         // Total Ne ejected during AGB (Msun)
    double Mg_yield;         // Total Mg ejected during AGB (Msun)

    double Z_yield_total;    // Total metals ejected during AGB (Msun)
    double M_lost;           // Total mass lost during AGB (Msun)
};


// IMF-integrated AGB ejecta for a single stellar population (SSP).
//
// These are NOT individual-star yields. Each value is the total AGB
// ejecta of that element per unit initial stellar-population mass:
//
//     y_X = M_X,ejected / M_stars,initial
//
// Units are therefore Msun / Msun (dimensionless mass fractions).
struct AGBIMFYield {
    double Z_init;

    double C_per_Mstar;
    double N_per_Mstar;
    double O_per_Mstar;
    double Ne_per_Mstar;
    double Mg_per_Mstar;

    double Z_per_Mstar;
};


class AGBYieldTable {
private:
    std::vector<AGBYieldEntry> table;
    std::vector<double> mass_grid;
    std::vector<double> Z_grid;

    // IMF-integrated SSP yields, tabulated as a function of metallicity.
    std::vector<AGBIMFYield> imf_yields;

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

    // Load MESA yield table from file.
    bool load_from_file(const char* filename);

    // Interpolate the individual-star MESA yield grid at a specified
    // initial stellar mass and metallicity.
    AGBYieldEntry interpolate_yields(double mass_msun, double Z_star);

    // Return IMF-integrated AGB ejecta per unit stellar-population mass
    // at the requested initial stellar metallicity.
    AGBIMFYield get_imf_yields(double Z_star);

    // Convenience getters.
    double get_total_metal_yield(double mass_msun, double Z_star);
    double get_C_yield(double mass_msun, double Z_star);
    double get_N_yield(double mass_msun, double Z_star);
    double get_O_yield(double mass_msun, double Z_star);

    // Total mass lost during the AGB phase.
    double get_mass_lost(double mass_msun, double Z_star);

    // Check whether table has been loaded.
    bool is_table_loaded() const { return is_loaded; }

    // Diagnostics.
    void print_diagnostics() const;
    void reset_diagnostics();
    int get_lookup_count() const { return lookup_count; }


private:
    // Locate interpolation brackets in mass and metallicity grids.
    void find_mass_bracket(double mass,
                           int& i_low, int& i_high,
                           double& frac);

    void find_Z_bracket(double Z,
                        int& i_low, int& i_high,
                        double& frac);

    // Build the IMF-integrated SSP yield table from the individual
    // MESA stellar models after the input table has been loaded.
    void build_imf_yield_table();

    // Kroupa IMF used to integrate the individual MESA models into
    // yields per unit mass of an entire stellar population.
    static double kroupa_imf(double mass);

    // Integrate M * phi(M) between two masses. Used to normalize
    // AGB ejecta by the total initial mass of the stellar population.
    static double integrate_imf_mass(double m1, double m2);
};


// Global instance.
extern AGBYieldTable AGB_Yields;

#endif // AGB_YIELDS_H