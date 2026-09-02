#include "gadgetconfig.h"

#include "snii_yields.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

SNIIYieldTable SNII_Yields;

namespace
{
    constexpr double SN_MASS_MIN = 13.0;
    constexpr double SN_MASS_MAX = 40.0;
    constexpr double HN_MASS_MIN = 20.0;

    // Keep the SSP normalization consistent with the AGB implementation.
    constexpr double IMF_NORM_MASS_MIN = 0.08;
    constexpr double IMF_NORM_MASS_MAX = 100.0;

    constexpr double ROW_TOL = 1.0e-8;
    constexpr double PHYS_TOL = 1.000001;

    inline double lerp(double y0, double y1, double f)
    {
        return y0 + f * (y1 - y0);
    }

    void find_bracket(const std::vector<double> &grid,
                    double x,
                    int &i_low,
                    int &i_high,
                    double &frac)
    {
        if(grid.empty())
        {
            i_low = i_high = -1;
            frac = 0.0;
            return;
        }

        if(x <= grid.front())
        {
            i_low = i_high = 0;
            frac = 0.0;
            return;
        }

        if(x >= grid.back())
        {
            i_low = i_high = static_cast<int>(grid.size()) - 1;
            frac = 0.0;
            return;
        }

        auto it = std::lower_bound(grid.begin(), grid.end(), x);
        i_high = static_cast<int>(it - grid.begin());
        i_low = i_high - 1;

        const double x0 = grid[i_low];
        const double x1 = grid[i_high];

        frac = (x1 > x0) ? (x - x0) / (x1 - x0) : 0.0;
    }

    const SNIIYieldEntry *find_sn_entry(const std::vector<SNIIYieldEntry> &table,
                                        double mass,
                                        double Z)
    {
        for(const auto &e : table)
        {
            if(std::fabs(e.mass_init - mass) < ROW_TOL &&
            std::fabs(e.Z_init - Z) < ROW_TOL)
                return &e;
        }

        return nullptr;
    }

    const HypernovaYieldEntry *find_hn_entry(const std::vector<HypernovaYieldEntry> &table,
                                            double mass,
                                            double Z)
    {
        for(const auto &e : table)
        {
            if(std::fabs(e.mass_init - mass) < ROW_TOL &&
            std::fabs(e.Z_init - Z) < ROW_TOL)
                return &e;
        }

        return nullptr;
    }

    bool contains_value(const std::vector<double> &grid, double x)
    {
        for(double v : grid)
            if(std::fabs(v - x) < ROW_TOL)
                return true;

        return false;
    }

    bool validate_sn_entry(const SNIIYieldEntry &e)
    {
        const double tracked =
            e.C_yield +
            e.N_yield +
            e.O_yield +
            e.Ne_yield +
            e.Mg_yield +
            e.Si_yield +
            e.Fe_yield;

        if(!std::isfinite(e.mass_init) ||
        !std::isfinite(e.Z_init) ||
        !std::isfinite(e.C_yield) ||
        !std::isfinite(e.N_yield) ||
        !std::isfinite(e.O_yield) ||
        !std::isfinite(e.Ne_yield) ||
        !std::isfinite(e.Mg_yield) ||
        !std::isfinite(e.Si_yield) ||
        !std::isfinite(e.Fe_yield) ||
        !std::isfinite(e.Z_yield_total) ||
        !std::isfinite(e.M_ejecta))
            return false;

        if(e.mass_init <= 0.0 ||
        e.Z_init < 0.0 ||
        e.C_yield < 0.0 ||
        e.N_yield < 0.0 ||
        e.O_yield < 0.0 ||
        e.Ne_yield < 0.0 ||
        e.Mg_yield < 0.0 ||
        e.Si_yield < 0.0 ||
        e.Fe_yield < 0.0 ||
        e.Z_yield_total < 0.0 ||
        e.M_ejecta < 0.0)
            return false;

        if(tracked > e.Z_yield_total * PHYS_TOL)
            return false;

        if(e.Z_yield_total > e.M_ejecta * PHYS_TOL)
            return false;

        return true;
    }

    bool validate_hn_entry(const HypernovaYieldEntry &e)
    {
        const double tracked =
            e.C_yield +
            e.N_yield +
            e.O_yield +
            e.Ne_yield +
            e.Mg_yield +
            e.Si_yield +
            e.Fe_yield;

        if(!std::isfinite(e.mass_init) ||
        !std::isfinite(e.Z_init) ||
        !std::isfinite(e.E51) ||
        !std::isfinite(e.C_yield) ||
        !std::isfinite(e.N_yield) ||
        !std::isfinite(e.O_yield) ||
        !std::isfinite(e.Ne_yield) ||
        !std::isfinite(e.Mg_yield) ||
        !std::isfinite(e.Si_yield) ||
        !std::isfinite(e.Fe_yield) ||
        !std::isfinite(e.Z_yield_total) ||
        !std::isfinite(e.M_ejecta))
            return false;

        if(e.mass_init <= 0.0 ||
        e.Z_init < 0.0 ||
        e.E51 <= 0.0 ||
        e.C_yield < 0.0 ||
        e.N_yield < 0.0 ||
        e.O_yield < 0.0 ||
        e.Ne_yield < 0.0 ||
        e.Mg_yield < 0.0 ||
        e.Si_yield < 0.0 ||
        e.Fe_yield < 0.0 ||
        e.Z_yield_total < 0.0 ||
        e.M_ejecta < 0.0)
            return false;

        if(tracked > e.Z_yield_total * PHYS_TOL)
            return false;

        if(e.Z_yield_total > e.M_ejecta * PHYS_TOL)
            return false;

        return true;
    }

    struct EffectiveYield
    {
        double C = 0.0;
        double N = 0.0;
        double O = 0.0;
        double Ne = 0.0;
        double Mg = 0.0;
        double Si = 0.0;
        double Fe = 0.0;
        double Z = 0.0;
        double E51 = 0.0;
    };

} // namespace


SNIIYieldTable::SNIIYieldTable()
{
    sn_loaded = false;
    hn_loaded = false;

    // Fiducial Kobayashi-style choice. This can be exposed as a
    // runtime parameter later.
    hypernova_fraction = 0.5;
}


bool SNIIYieldTable::load_sn_from_file(const char *filename)
{
    sn_table.clear();
    sn_mass_grid.clear();
    Z_grid.clear();
    imf_yields.clear();

    sn_loaded = false;

    std::ifstream infile(filename);

    if(!infile.is_open())
    {
        printf("[SNII_YIELDS] ERROR: could not open SN yield file: %s\n",
               filename);
        return false;
    }

    std::string line;
    int line_number = 0;

    while(std::getline(infile, line))
    {
        line_number++;

        const std::size_t first = line.find_first_not_of(" \t\r\n");

        if(first == std::string::npos)
            continue;

        if(line[first] == '#')
            continue;

        std::istringstream iss(line);

        SNIIYieldEntry entry;

        if(!(iss >> entry.mass_init
                 >> entry.Z_init
                 >> entry.C_yield
                 >> entry.N_yield
                 >> entry.O_yield
                 >> entry.Ne_yield
                 >> entry.Mg_yield
                 >> entry.Si_yield
                 >> entry.Fe_yield
                 >> entry.Z_yield_total
                 >> entry.M_ejecta))
        {
            printf("[SNII_YIELDS] ERROR: malformed SN row at line %d:\n%s\n",
                   line_number, line.c_str());
            return false;
        }

        if(!validate_sn_entry(entry))
        {
            printf("[SNII_YIELDS] ERROR: invalid SN yield row at line %d:\n%s\n",
                   line_number, line.c_str());
            return false;
        }

        sn_table.push_back(entry);

        if(!contains_value(sn_mass_grid, entry.mass_init))
            sn_mass_grid.push_back(entry.mass_init);

        if(!contains_value(Z_grid, entry.Z_init))
            Z_grid.push_back(entry.Z_init);
    }

    infile.close();

    std::sort(sn_mass_grid.begin(), sn_mass_grid.end());
    std::sort(Z_grid.begin(), Z_grid.end());

    if(sn_table.empty())
    {
        printf("[SNII_YIELDS] ERROR: no valid SN yields loaded from %s\n",
               filename);
        return false;
    }

    const std::size_t expected_rows =
        sn_mass_grid.size() * Z_grid.size();

    if(sn_table.size() != expected_rows)
    {
        printf("[SNII_YIELDS] ERROR: SN grid is incomplete: "
               "%zu rows loaded, but %zu are expected from "
               "%zu masses x %zu metallicities\n",
               sn_table.size(),
               expected_rows,
               sn_mass_grid.size(),
               Z_grid.size());
        return false;
    }

    for(double Z : Z_grid)
    {
        for(double mass : sn_mass_grid)
        {
            if(find_sn_entry(sn_table, mass, Z) == nullptr)
            {
                printf("[SNII_YIELDS] ERROR: missing SN model "
                       "M=%g Msun, Z=%g\n",
                       mass, Z);
                return false;
            }
        }
    }

    sn_loaded = true;

    printf("[SNII_YIELDS] Loaded %zu ordinary SN models from %s\n",
           sn_table.size(), filename);

    printf("[SNII_YIELDS] SN grid: %zu masses x %zu metallicities\n",
           sn_mass_grid.size(), Z_grid.size());

    return true;
}


bool SNIIYieldTable::load_hn_from_file(const char *filename)
{
    hn_table.clear();
    hn_mass_grid.clear();
    imf_yields.clear();

    hn_loaded = false;

    if(!sn_loaded)
    {
        printf("[SNII_YIELDS] ERROR: ordinary SN table must be loaded "
               "before the HN table\n");
        return false;
    }

    std::ifstream infile(filename);

    if(!infile.is_open())
    {
        printf("[SNII_YIELDS] ERROR: could not open HN yield file: %s\n",
               filename);
        return false;
    }

    std::string line;
    int line_number = 0;

    while(std::getline(infile, line))
    {
        line_number++;

        const std::size_t first = line.find_first_not_of(" \t\r\n");

        if(first == std::string::npos)
            continue;

        if(line[first] == '#')
            continue;

        std::istringstream iss(line);

        HypernovaYieldEntry entry;

        if(!(iss >> entry.mass_init
                 >> entry.Z_init
                 >> entry.E51
                 >> entry.C_yield
                 >> entry.N_yield
                 >> entry.O_yield
                 >> entry.Ne_yield
                 >> entry.Mg_yield
                 >> entry.Si_yield
                 >> entry.Fe_yield
                 >> entry.Z_yield_total
                 >> entry.M_ejecta))
        {
            printf("[SNII_YIELDS] ERROR: malformed HN row at line %d:\n%s\n",
                   line_number, line.c_str());
            return false;
        }

        if(!validate_hn_entry(entry))
        {
            printf("[SNII_YIELDS] ERROR: invalid HN yield row at line %d:\n%s\n",
                   line_number, line.c_str());
            return false;
        }

        if(!contains_value(Z_grid, entry.Z_init))
        {
            printf("[SNII_YIELDS] ERROR: HN metallicity Z=%g at line %d "
                   "is not present in the ordinary SN grid\n",
                   entry.Z_init, line_number);
            return false;
        }

        hn_table.push_back(entry);

        if(!contains_value(hn_mass_grid, entry.mass_init))
            hn_mass_grid.push_back(entry.mass_init);
    }

    infile.close();

    std::sort(hn_mass_grid.begin(), hn_mass_grid.end());

    if(hn_table.empty())
    {
        printf("[SNII_YIELDS] ERROR: no valid HN yields loaded from %s\n",
               filename);
        return false;
    }

    const std::size_t expected_rows =
        hn_mass_grid.size() * Z_grid.size();

    if(hn_table.size() != expected_rows)
    {
        printf("[SNII_YIELDS] ERROR: HN grid is incomplete: "
               "%zu rows loaded, but %zu are expected from "
               "%zu masses x %zu metallicities\n",
               hn_table.size(),
               expected_rows,
               hn_mass_grid.size(),
               Z_grid.size());
        return false;
    }

    for(double Z : Z_grid)
    {
        for(double mass : hn_mass_grid)
        {
            if(find_hn_entry(hn_table, mass, Z) == nullptr)
            {
                printf("[SNII_YIELDS] ERROR: missing HN model "
                       "M=%g Msun, Z=%g\n",
                       mass, Z);
                return false;
            }
        }
    }

    hn_loaded = true;

    printf("[SNII_YIELDS] Loaded %zu hypernova models from %s\n",
           hn_table.size(), filename);

    printf("[SNII_YIELDS] HN grid: %zu masses x %zu metallicities\n",
           hn_mass_grid.size(), Z_grid.size());

    return true;
}


void SNIIYieldTable::set_hypernova_fraction(double f)
{
    if(!std::isfinite(f))
    {
        printf("[SNII_YIELDS] WARNING: non-finite HN fraction ignored\n");
        return;
    }

    if(f < 0.0)
        f = 0.0;

    if(f > 1.0)
        f = 1.0;

    hypernova_fraction = f;

    // The effective SSP yields depend on this parameter.
    imf_yields.clear();
}


SNIIYieldEntry SNIIYieldTable::interpolate_sn(double mass, double Z) const
{
    SNIIYieldEntry result {};

    if(!sn_loaded || sn_table.empty() ||
       sn_mass_grid.empty() || Z_grid.empty())
        return result;

    int im0, im1, iz0, iz1;
    double fm, fz;

    find_bracket(sn_mass_grid, mass, im0, im1, fm);
    find_bracket(Z_grid, Z, iz0, iz1, fz);

    const double m0 = sn_mass_grid[im0];
    const double m1 = sn_mass_grid[im1];
    const double z0 = Z_grid[iz0];
    const double z1 = Z_grid[iz1];

    const SNIIYieldEntry *e00 = find_sn_entry(sn_table, m0, z0);
    const SNIIYieldEntry *e10 = find_sn_entry(sn_table, m1, z0);
    const SNIIYieldEntry *e01 = find_sn_entry(sn_table, m0, z1);
    const SNIIYieldEntry *e11 = find_sn_entry(sn_table, m1, z1);

    if(e00 == nullptr || e10 == nullptr ||
       e01 == nullptr || e11 == nullptr)
        return result;

    auto bilerp = [fm, fz](double q00, double q10,
                           double q01, double q11)
    {
        const double qz0 = lerp(q00, q10, fm);
        const double qz1 = lerp(q01, q11, fm);
        return lerp(qz0, qz1, fz);
    };

    result.mass_init = mass;
    result.Z_init = Z;

    result.C_yield =
        bilerp(e00->C_yield, e10->C_yield,
               e01->C_yield, e11->C_yield);

    result.N_yield =
        bilerp(e00->N_yield, e10->N_yield,
               e01->N_yield, e11->N_yield);

    result.O_yield =
        bilerp(e00->O_yield, e10->O_yield,
               e01->O_yield, e11->O_yield);

    result.Ne_yield =
        bilerp(e00->Ne_yield, e10->Ne_yield,
               e01->Ne_yield, e11->Ne_yield);

    result.Mg_yield =
        bilerp(e00->Mg_yield, e10->Mg_yield,
               e01->Mg_yield, e11->Mg_yield);

    result.Si_yield =
        bilerp(e00->Si_yield, e10->Si_yield,
               e01->Si_yield, e11->Si_yield);

    result.Fe_yield =
        bilerp(e00->Fe_yield, e10->Fe_yield,
               e01->Fe_yield, e11->Fe_yield);

    result.Z_yield_total =
        bilerp(e00->Z_yield_total, e10->Z_yield_total,
               e01->Z_yield_total, e11->Z_yield_total);

    result.M_ejecta =
        bilerp(e00->M_ejecta, e10->M_ejecta,
               e01->M_ejecta, e11->M_ejecta);

    return result;
}


HypernovaYieldEntry SNIIYieldTable::interpolate_hn(double mass, double Z) const
{
    HypernovaYieldEntry result {};

    if(!hn_loaded || hn_table.empty() ||
       hn_mass_grid.empty() || Z_grid.empty())
        return result;

    int im0, im1, iz0, iz1;
    double fm, fz;

    find_bracket(hn_mass_grid, mass, im0, im1, fm);
    find_bracket(Z_grid, Z, iz0, iz1, fz);

    const double m0 = hn_mass_grid[im0];
    const double m1 = hn_mass_grid[im1];
    const double z0 = Z_grid[iz0];
    const double z1 = Z_grid[iz1];

    const HypernovaYieldEntry *e00 = find_hn_entry(hn_table, m0, z0);
    const HypernovaYieldEntry *e10 = find_hn_entry(hn_table, m1, z0);
    const HypernovaYieldEntry *e01 = find_hn_entry(hn_table, m0, z1);
    const HypernovaYieldEntry *e11 = find_hn_entry(hn_table, m1, z1);

    if(e00 == nullptr || e10 == nullptr ||
       e01 == nullptr || e11 == nullptr)
        return result;

    auto bilerp = [fm, fz](double q00, double q10,
                           double q01, double q11)
    {
        const double qz0 = lerp(q00, q10, fm);
        const double qz1 = lerp(q01, q11, fm);
        return lerp(qz0, qz1, fz);
    };

    result.mass_init = mass;
    result.Z_init = Z;

    result.E51 =
        bilerp(e00->E51, e10->E51,
               e01->E51, e11->E51);

    result.C_yield =
        bilerp(e00->C_yield, e10->C_yield,
               e01->C_yield, e11->C_yield);

    result.N_yield =
        bilerp(e00->N_yield, e10->N_yield,
               e01->N_yield, e11->N_yield);

    result.O_yield =
        bilerp(e00->O_yield, e10->O_yield,
               e01->O_yield, e11->O_yield);

    result.Ne_yield =
        bilerp(e00->Ne_yield, e10->Ne_yield,
               e01->Ne_yield, e11->Ne_yield);

    result.Mg_yield =
        bilerp(e00->Mg_yield, e10->Mg_yield,
               e01->Mg_yield, e11->Mg_yield);

    result.Si_yield =
        bilerp(e00->Si_yield, e10->Si_yield,
               e01->Si_yield, e11->Si_yield);

    result.Fe_yield =
        bilerp(e00->Fe_yield, e10->Fe_yield,
               e01->Fe_yield, e11->Fe_yield);

    result.Z_yield_total =
        bilerp(e00->Z_yield_total, e10->Z_yield_total,
               e01->Z_yield_total, e11->Z_yield_total);

    result.M_ejecta =
        bilerp(e00->M_ejecta, e10->M_ejecta,
               e01->M_ejecta, e11->M_ejecta);

    return result;
}


double SNIIYieldTable::kroupa_imf(double mass)
{
    if(mass <= 0.0)
        return 0.0;

    // Kroupa-like two-part IMF, with continuity at 0.5 Msun.
    if(mass < 0.5)
        return std::pow(mass, -1.3);

    return 0.5 * std::pow(mass, -2.3);
}


double SNIIYieldTable::integrate_imf_mass(double m1, double m2)
{
    if(m2 <= m1 || m2 <= 0.0)
        return 0.0;

    if(m1 < 0.0)
        m1 = 0.0;

    double result = 0.0;

    // Below 0.5 Msun:
    // M * phi(M) = M^-0.3
    if(m1 < 0.5)
    {
        const double lo = m1;
        const double hi = std::min(m2, 0.5);

        if(hi > lo)
            result +=
                (std::pow(hi, 0.7) -
                 std::pow(lo, 0.7)) / 0.7;
    }

    // Above 0.5 Msun:
    // M * phi(M) = 0.5 M^-1.3
    if(m2 > 0.5)
    {
        const double lo = std::max(m1, 0.5);
        const double hi = m2;

        if(hi > lo)
            result +=
                0.5 *
                (std::pow(lo, -0.3) -
                 std::pow(hi, -0.3)) / 0.3;
    }

    return result;
}


void SNIIYieldTable::build_imf_yield_table()
{
    imf_yields.clear();

    if(!sn_loaded || !hn_loaded)
    {
        printf("[SNII_YIELDS] ERROR: cannot build IMF table until both "
               "SN and HN tables are loaded\n");
        return;
    }

    if(sn_mass_grid.size() < 2 || Z_grid.empty())
    {
        printf("[SNII_YIELDS] ERROR: insufficient SN grid to build IMF table\n");
        return;
    }

    const double imf_mass_norm =
        integrate_imf_mass(IMF_NORM_MASS_MIN,
                           IMF_NORM_MASS_MAX);

    if(imf_mass_norm <= 0.0 ||
       !std::isfinite(imf_mass_norm))
    {
        printf("[SNII_YIELDS] ERROR: invalid IMF mass normalization\n");
        return;
    }

    printf("[SNII_YIELDS] Building IMF-integrated SNII+HN yields\n");
    printf("[SNII_YIELDS] HN fraction = %.3f\n",
           hypernova_fraction);
    printf("[SNII_YIELDS] Yield-supported progenitor range = "
           "%.1f-%.1f Msun\n",
           SN_MASS_MIN, SN_MASS_MAX);
    printf("[SNII_YIELDS] SSP IMF normalization range = "
           "%.2f-%.1f Msun\n",
           IMF_NORM_MASS_MIN, IMF_NORM_MASS_MAX);

    for(double Z : Z_grid)
    {
        double int_C = 0.0;
        double int_N = 0.0;
        double int_O = 0.0;
        double int_Ne = 0.0;
        double int_Mg = 0.0;
        double int_Si = 0.0;
        double int_Fe = 0.0;
        double int_Z = 0.0;
        double int_E51 = 0.0;

        // Integrate only over the progenitor-mass range directly covered
        // by the Kobayashi ordinary-SN table. We do NOT extrapolate yields
        // below 13 or above 40 Msun.
        for(std::size_t i = 0;
            i + 1 < sn_mass_grid.size();
            ++i)
        {
            const double m0 = sn_mass_grid[i];
            const double m1 = sn_mass_grid[i + 1];

            if(m1 <= SN_MASS_MIN ||
               m0 >= SN_MASS_MAX)
                continue;

            SNIIYieldEntry sn0 = interpolate_sn(m0, Z);
            SNIIYieldEntry sn1 = interpolate_sn(m1, Z);

            EffectiveYield y0;
            EffectiveYield y1;

            // Left endpoint.
            y0.C = sn0.C_yield;
            y0.N = sn0.N_yield;
            y0.O = sn0.O_yield;
            y0.Ne = sn0.Ne_yield;
            y0.Mg = sn0.Mg_yield;
            y0.Si = sn0.Si_yield;
            y0.Fe = sn0.Fe_yield;
            y0.Z = sn0.Z_yield_total;
            y0.E51 = 1.0;

            // Right endpoint.
            y1.C = sn1.C_yield;
            y1.N = sn1.N_yield;
            y1.O = sn1.O_yield;
            y1.Ne = sn1.Ne_yield;
            y1.Mg = sn1.Mg_yield;
            y1.Si = sn1.Si_yield;
            y1.Fe = sn1.Fe_yield;
            y1.Z = sn1.Z_yield_total;
            y1.E51 = 1.0;

            // HN models begin at 20 Msun. To avoid artificially ramping
            // the HN contribution through the 18-20 Msun interval, the
            // [18,20] interval remains ordinary-SN-only. Starting with
            // the [20,25] interval, both endpoints use the mixed SN/HN
            // prescription.
            if(m0 >= HN_MASS_MIN)
            {
                HypernovaYieldEntry hn0 =
                    interpolate_hn(m0, Z);

                HypernovaYieldEntry hn1 =
                    interpolate_hn(m1, Z);

                const double fsn = 1.0 - hypernova_fraction;
                const double fhn = hypernova_fraction;

                y0.C = fsn * sn0.C_yield +
                       fhn * hn0.C_yield;
                y0.N = fsn * sn0.N_yield +
                       fhn * hn0.N_yield;
                y0.O = fsn * sn0.O_yield +
                       fhn * hn0.O_yield;
                y0.Ne = fsn * sn0.Ne_yield +
                        fhn * hn0.Ne_yield;
                y0.Mg = fsn * sn0.Mg_yield +
                        fhn * hn0.Mg_yield;
                y0.Si = fsn * sn0.Si_yield +
                        fhn * hn0.Si_yield;
                y0.Fe = fsn * sn0.Fe_yield +
                        fhn * hn0.Fe_yield;
                y0.Z = fsn * sn0.Z_yield_total +
                       fhn * hn0.Z_yield_total;

                y0.E51 =
                    fsn * 1.0 +
                    fhn * hn0.E51;

                y1.C = fsn * sn1.C_yield +
                       fhn * hn1.C_yield;
                y1.N = fsn * sn1.N_yield +
                       fhn * hn1.N_yield;
                y1.O = fsn * sn1.O_yield +
                       fhn * hn1.O_yield;
                y1.Ne = fsn * sn1.Ne_yield +
                        fhn * hn1.Ne_yield;
                y1.Mg = fsn * sn1.Mg_yield +
                        fhn * hn1.Mg_yield;
                y1.Si = fsn * sn1.Si_yield +
                        fhn * hn1.Si_yield;
                y1.Fe = fsn * sn1.Fe_yield +
                        fhn * hn1.Fe_yield;
                y1.Z = fsn * sn1.Z_yield_total +
                       fhn * hn1.Z_yield_total;

                y1.E51 =
                    fsn * 1.0 +
                    fhn * hn1.E51;
            }

            const double phi0 = kroupa_imf(m0);
            const double phi1 = kroupa_imf(m1);
            const double dm = m1 - m0;

            int_C +=
                0.5 * (phi0 * y0.C +
                       phi1 * y1.C) * dm;

            int_N +=
                0.5 * (phi0 * y0.N +
                       phi1 * y1.N) * dm;

            int_O +=
                0.5 * (phi0 * y0.O +
                       phi1 * y1.O) * dm;

            int_Ne +=
                0.5 * (phi0 * y0.Ne +
                       phi1 * y1.Ne) * dm;

            int_Mg +=
                0.5 * (phi0 * y0.Mg +
                       phi1 * y1.Mg) * dm;

            int_Si +=
                0.5 * (phi0 * y0.Si +
                       phi1 * y1.Si) * dm;

            int_Fe +=
                0.5 * (phi0 * y0.Fe +
                       phi1 * y1.Fe) * dm;

            int_Z +=
                0.5 * (phi0 * y0.Z +
                       phi1 * y1.Z) * dm;

            // Explosion energy is per exploding star, not a mass yield.
            // Therefore integrate phi(M) * E51(M), then divide by the
            // total SSP initial stellar mass.
            int_E51 +=
                0.5 * (phi0 * y0.E51 +
                       phi1 * y1.E51) * dm;
        }

        SNIIIMFYield y {};

        y.Z_init = Z;

        y.C_per_Mstar = int_C / imf_mass_norm;
        y.N_per_Mstar = int_N / imf_mass_norm;
        y.O_per_Mstar = int_O / imf_mass_norm;
        y.Ne_per_Mstar = int_Ne / imf_mass_norm;
        y.Mg_per_Mstar = int_Mg / imf_mass_norm;
        y.Si_per_Mstar = int_Si / imf_mass_norm;
        y.Fe_per_Mstar = int_Fe / imf_mass_norm;

        y.Z_per_Mstar = int_Z / imf_mass_norm;
        y.E51_per_Mstar = int_E51 / imf_mass_norm;

        const double tracked =
            y.C_per_Mstar +
            y.N_per_Mstar +
            y.O_per_Mstar +
            y.Ne_per_Mstar +
            y.Mg_per_Mstar +
            y.Si_per_Mstar +
            y.Fe_per_Mstar;

        if(!std::isfinite(y.Z_per_Mstar) ||
           !std::isfinite(y.C_per_Mstar) ||
           !std::isfinite(y.N_per_Mstar) ||
           !std::isfinite(y.O_per_Mstar) ||
           !std::isfinite(y.Ne_per_Mstar) ||
           !std::isfinite(y.Mg_per_Mstar) ||
           !std::isfinite(y.Si_per_Mstar) ||
           !std::isfinite(y.Fe_per_Mstar) ||
           !std::isfinite(y.E51_per_Mstar))
        {
            printf("[SNII_YIELDS] ERROR: non-finite IMF-integrated yield "
                   "at Z=%g\n", Z);
            imf_yields.clear();
            return;
        }

        if(y.Z_per_Mstar < 0.0 ||
           y.C_per_Mstar < 0.0 ||
           y.N_per_Mstar < 0.0 ||
           y.O_per_Mstar < 0.0 ||
           y.Ne_per_Mstar < 0.0 ||
           y.Mg_per_Mstar < 0.0 ||
           y.Si_per_Mstar < 0.0 ||
           y.Fe_per_Mstar < 0.0 ||
           y.E51_per_Mstar < 0.0)
        {
            printf("[SNII_YIELDS] ERROR: negative IMF-integrated yield "
                   "at Z=%g\n", Z);
            imf_yields.clear();
            return;
        }

        if(tracked > y.Z_per_Mstar * PHYS_TOL)
        {
            printf("[SNII_YIELDS] ERROR: tracked IMF-integrated elements "
                   "exceed total Z at Z=%g\n", Z);
            imf_yields.clear();
            return;
        }

        imf_yields.push_back(y);

        printf("[SNII_YIELDS] Z=%7.4f  "
               "yZ=%10.4e  "
               "yC=%10.4e  "
               "yN=%10.4e  "
               "yO=%10.4e  "
               "yNe=%10.4e  "
               "yMg=%10.4e  "
               "ySi=%10.4e  "
               "yFe=%10.4e  "
               "E51/Mstar=%10.4e\n",
               y.Z_init,
               y.Z_per_Mstar,
               y.C_per_Mstar,
               y.N_per_Mstar,
               y.O_per_Mstar,
               y.Ne_per_Mstar,
               y.Mg_per_Mstar,
               y.Si_per_Mstar,
               y.Fe_per_Mstar,
               y.E51_per_Mstar);
    }

    std::sort(imf_yields.begin(),
              imf_yields.end(),
              [](const SNIIIMFYield &a,
                 const SNIIIMFYield &b)
              {
                  return a.Z_init < b.Z_init;
              });

    printf("[SNII_YIELDS] Built IMF-integrated SSP yield table "
           "for %zu metallicities\n",
           imf_yields.size());
}


SNIIIMFYield SNIIYieldTable::get_imf_yields(double Z_star) const
{
    SNIIIMFYield empty {};

    if(imf_yields.empty())
        return empty;

    if(Z_star <= imf_yields.front().Z_init)
        return imf_yields.front();

    if(Z_star >= imf_yields.back().Z_init)
        return imf_yields.back();

    for(std::size_t i = 0;
        i + 1 < imf_yields.size();
        ++i)
    {
        const SNIIIMFYield &lo = imf_yields[i];
        const SNIIIMFYield &hi = imf_yields[i + 1];

        if(Z_star >= lo.Z_init &&
           Z_star <= hi.Z_init)
        {
            const double dz =
                hi.Z_init - lo.Z_init;

            const double f =
                (dz > 0.0) ?
                (Z_star - lo.Z_init) / dz :
                0.0;

            SNIIIMFYield result {};

            result.Z_init = Z_star;

            result.C_per_Mstar =
                lerp(lo.C_per_Mstar,
                     hi.C_per_Mstar, f);

            result.N_per_Mstar =
                lerp(lo.N_per_Mstar,
                     hi.N_per_Mstar, f);

            result.O_per_Mstar =
                lerp(lo.O_per_Mstar,
                     hi.O_per_Mstar, f);

            result.Ne_per_Mstar =
                lerp(lo.Ne_per_Mstar,
                     hi.Ne_per_Mstar, f);

            result.Mg_per_Mstar =
                lerp(lo.Mg_per_Mstar,
                     hi.Mg_per_Mstar, f);

            result.Si_per_Mstar =
                lerp(lo.Si_per_Mstar,
                     hi.Si_per_Mstar, f);

            result.Fe_per_Mstar =
                lerp(lo.Fe_per_Mstar,
                     hi.Fe_per_Mstar, f);

            result.Z_per_Mstar =
                lerp(lo.Z_per_Mstar,
                     hi.Z_per_Mstar, f);

            result.E51_per_Mstar =
                lerp(lo.E51_per_Mstar,
                     hi.E51_per_Mstar, f);

            return result;
        }
    }

    return empty;
}