#ifndef SNII_YIELDS_H
#define SNII_YIELDS_H

#include "gadgetconfig.h"

#include <vector>

struct SNIIYieldEntry
{
    double mass_init;
    double Z_init;

    double C_yield;
    double N_yield;
    double O_yield;
    double Ne_yield;
    double Mg_yield;
    double Si_yield;
    double Fe_yield;

    double Z_yield_total;
    double M_ejecta;
};


struct HypernovaYieldEntry
{
    double mass_init;
    double Z_init;
    double E51;

    double C_yield;
    double N_yield;
    double O_yield;
    double Ne_yield;
    double Mg_yield;
    double Si_yield;
    double Fe_yield;

    double Z_yield_total;
    double M_ejecta;
};


// IMF-integrated effective massive-star ejecta for one SSP.
struct SNIIIMFYield
{
    double Z_init;

    double C_per_Mstar;
    double N_per_Mstar;
    double O_per_Mstar;
    double Ne_per_Mstar;
    double Mg_per_Mstar;
    double Si_per_Mstar;
    double Fe_per_Mstar;

    double Z_per_Mstar;

    // IMF-averaged explosion energy per unit initial SSP mass.
    // Units: 1e51 erg / Msun formed.
    double E51_per_Mstar;
};


class SNIIYieldTable
{
private:
    std::vector<SNIIYieldEntry> sn_table;
    std::vector<HypernovaYieldEntry> hn_table;

    std::vector<double> sn_mass_grid;
    std::vector<double> hn_mass_grid;
    std::vector<double> Z_grid;

    std::vector<SNIIIMFYield> imf_yields;

    bool sn_loaded;
    bool hn_loaded;

    double hypernova_fraction;

public:
    SNIIYieldTable();

    bool load_sn_from_file(const char *filename);
    bool load_hn_from_file(const char *filename);

    // Build the effective SNII+HN SSP table after both files are loaded.
    void build_imf_yield_table();

    SNIIIMFYield get_imf_yields(double Z_star) const;

    bool is_loaded() const
    {
        return sn_loaded && hn_loaded;
    }

    void set_hypernova_fraction(double f);
    double get_hypernova_fraction() const
    {
        return hypernova_fraction;
    }

private:
    SNIIYieldEntry interpolate_sn(double mass, double Z) const;
    HypernovaYieldEntry interpolate_hn(double mass, double Z) const;

    static double kroupa_imf(double mass);
    static double integrate_imf_mass(double m1, double m2);
};


// Global instance.
extern SNIIYieldTable SNII_Yields;

#endif