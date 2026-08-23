"""Source-independent conversion of coupled surface mass fluxes.

Input providers are responsible for aggregating freshwater components, temporal
sampling, and remapping to the local Veros T grid. This module only converts
those inputs to the salinity flux consumed by the tracer integrator.
"""

from veros import KernelOutput, veros_kernel
from veros.core import utilities


@veros_kernel
def convert_surface_freshwater_flux(
    freshwater_flux,
    salt_flux,
    surface_mask,
    reference_salinity,
    freshwater_density,
    reference_density,
):
    """Convert surface mass fluxes to a virtual salinity flux.

    Freshwater and direct salt mass fluxes use ``kg / m^2 / s`` and are
    positive into the ocean. The returned salinity flux uses
    ``m (g / kg) / s``.
    """
    return surface_mask * (
        -reference_salinity * freshwater_flux / freshwater_density + 1000.0 * salt_flux / reference_density
    )


@veros_kernel
def set_surface_freshwater_flux(state):
    """Derive Veros' surface salinity flux from provider-owned mass fluxes.

    When coupled surface freshwater forcing is disabled, the setup-provided
    salinity flux is returned unchanged.
    """
    vs = state.variables
    settings = state.settings

    if not settings.enable_surface_freshwater_flux:
        return KernelOutput(forc_salt_surface=vs.forc_salt_surface)

    forc_salt_surface = convert_surface_freshwater_flux(
        vs.surface_freshwater_flux,
        vs.surface_salt_flux,
        vs.maskT[:, :, -1],
        settings.surface_salinity_reference,
        settings.rho_freshwater,
        settings.rho_0,
    )
    forc_salt_surface = utilities.enforce_boundaries(forc_salt_surface, settings.enable_cyclic_x)

    return KernelOutput(forc_salt_surface=forc_salt_surface)
