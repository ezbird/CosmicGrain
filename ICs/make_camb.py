import camb
from camb import model
import numpy as np

# ===== YOUR GADGET-4/MUSIC COSMOLOGY =====
h = 0.6732
Omega_m = 0.3158
Omega_Lambda = 0.6842
Omega_b = 0.04936
Omega_c = Omega_m - Omega_b
H0 = 100.0 * h

ombh2 = Omega_b * h**2
omch2 = Omega_c * h**2

sigma_8 = 0.8120
ns = 0.9665
zstart = 99

# ===== SETUP CAMB =====
pars = camb.CAMBparams()
pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=0.06, omk=0)
pars.InitPower.set_params(As=2.1e-9, ns=ns, r=0)
pars.set_matter_power(redshifts=[zstart], kmax=2000.0)
pars.NonLinear = model.NonLinear_none

pars.WantCls = False
pars.Want_CMB = False
pars.Want_transfer = True

# Calculate
results = camb.get_results(pars)

# ===== GET TRANSFER FUNCTIONS =====
trans = results.get_matter_transfer_data()
kh = trans.q  # k in h/Mpc

# Get individual transfer functions at z=zstart (index 0)
T_cdm = trans.transfer_z('delta_cdm', z_index=0)
T_baryon = trans.transfer_z('delta_baryon', z_index=0)
T_photon = trans.transfer_z('delta_photon', z_index=0)
T_nu = trans.transfer_z('delta_nu', z_index=0)
T_tot = trans.transfer_z('delta_tot', z_index=0)

# ===== WRITE IN CAMB FORMAT =====
# Standard CAMB format: k/h, T_cdm, T_baryon, T_photon, T_nu, T_nu_massive, T_total
output_file = 'camb_z99_transfer.dat'

with open(output_file, 'w') as f:
    f.write(f"# Transfer functions at z={zstart}\n")
    f.write("# k/h[Mpc^-1]  T_cdm  T_baryon  T_photon  T_nu  T_total\n")
    for i in range(len(kh)):
        f.write(f"{kh[i]:14.7e} {T_cdm[i]:14.7e} {T_baryon[i]:14.7e} {T_photon[i]:14.7e} {T_nu[i]:14.7e} {T_tot[i]:14.7e}\n")

print(f"Transfer function written to: {output_file}")

# Verify sigma_8
sigma8 = results.get_sigma8()[0]
print(f"sigma_8(z={zstart}) = {sigma8:.4f}")

print(f"\nUse in MUSIC2 config:")
print(f"transfer = camb_file")
print(f"transfer_file = {output_file}")
