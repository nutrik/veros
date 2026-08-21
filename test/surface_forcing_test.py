import numpy as onp
import pytest


def _cesm_like_coupled_provider(precipitation, evaporation, melt, runoff, ice_runoff, sea_ice_salt):
    """Aggregate already-remapped CESM-like component fluxes on the Veros T grid.

    .. list-table:: CESM-to-Veros provider component mapping
       :header-rows: 1

       * - Provider component
         - Positive contribution
         - Negative contribution
         - Veros field
       * - Atmospheric precipitation
         - Rain/snow entering ocean
         - Not applicable
         - ``surface_freshwater_flux``
       * - Evaporation
         - Not applicable
         - Water leaving ocean
         - ``surface_freshwater_flux``
       * - Sea-ice melt/freezing
         - Meltwater entering ocean
         - Freezing removes water
         - ``surface_freshwater_flux``
       * - Liquid runoff
         - Land runoff entering ocean
         - Normally disallowed
         - ``surface_freshwater_flux``
       * - Frozen/ice runoff
         - Frozen runoff entering ocean
         - Normally disallowed
         - ``surface_freshwater_flux``
       * - Sea-ice salt exchange
         - Salt entering ocean
         - Salt retained by forming ice
         - ``surface_salt_flux``

    The five freshwater inputs correspond to POP's ``PREC_F``, ``EVAP_F``,
    ``MELT_F``, ``ROFF_F``, and ``IOFF_F``. Direct sea-ice salt exchange
    corresponds to ``SALT_F`` and stays separate because it is salt mass rather
    than water mass. All inputs use kg / m^2 / s and are positive into the
    ocean.
    """
    return {
        "surface_freshwater_flux": precipitation + evaporation + melt + runoff + ice_runoff,
        "surface_salt_flux": sea_ice_salt,
    }


@pytest.fixture
def diskless_mode():
    from veros import runtime_settings

    original = runtime_settings.diskless_mode
    object.__setattr__(runtime_settings, "diskless_mode", True)
    try:
        yield
    finally:
        object.__setattr__(runtime_settings, "diskless_mode", original)


def _make_state(enable_surface_freshwater_flux):
    from veros.settings import SETTINGS
    from veros.state import VerosState
    from veros.variables import DIM_TO_SHAPE_VAR, VARIABLES

    state = VerosState(VARIABLES, SETTINGS, DIM_TO_SHAPE_VAR.copy())
    try:
        with state.settings.unlock():
            state.settings.nx = 2
            state.settings.ny = 1
            state.settings.nz = 1
            state.settings.enable_surface_freshwater_flux = enable_surface_freshwater_flux
    except AttributeError:
        pytest.fail("surface freshwater flux settings are not implemented")

    state.initialize_variables()
    return state


def test_positive_freshwater_flux_produces_negative_virtual_salinity_flux():
    try:
        from veros.core.surface_forcing import convert_surface_freshwater_flux
    except ModuleNotFoundError:
        pytest.fail("surface freshwater flux conversion is not implemented")

    from veros.core.operators import numpy as npx

    freshwater_flux = npx.array([[2.0, -1.0]])
    salt_flux = npx.zeros_like(freshwater_flux)
    surface_mask = npx.ones_like(freshwater_flux)

    actual = convert_surface_freshwater_flux(
        freshwater_flux,
        salt_flux,
        surface_mask,
        reference_salinity=35.0,
        freshwater_density=1000.0,
        reference_density=1025.0,
    )

    onp.testing.assert_allclose(actual, [[-0.07, 0.035]])


def test_direct_salt_flux_uses_boussinesq_reference_density():
    from veros.core.surface_forcing import convert_surface_freshwater_flux
    from veros.core.operators import numpy as npx

    freshwater_flux = npx.zeros((1, 2))
    salt_flux = npx.array([[1.0, -2.0]])
    surface_mask = npx.ones_like(freshwater_flux)

    actual = convert_surface_freshwater_flux(
        freshwater_flux,
        salt_flux,
        surface_mask,
        reference_salinity=35.0,
        freshwater_density=1000.0,
        reference_density=1025.0,
    )

    onp.testing.assert_allclose(actual, [[0.975609756097561, -1.951219512195122]])


def test_surface_mask_removes_flux_over_land():
    from veros.core.surface_forcing import convert_surface_freshwater_flux
    from veros.core.operators import numpy as npx

    freshwater_flux = npx.ones((1, 2))
    salt_flux = npx.zeros_like(freshwater_flux)
    surface_mask = npx.array([[1.0, 0.0]])

    actual = convert_surface_freshwater_flux(
        freshwater_flux,
        salt_flux,
        surface_mask,
        reference_salinity=35.0,
        freshwater_density=1000.0,
        reference_density=1025.0,
    )

    onp.testing.assert_allclose(actual, [[-0.035, 0.0]])


def test_cesm_like_provider_supplies_separate_conditional_t_grid_fluxes():
    """Exercise provider aggregation and Veros conversion in one realistic toy case."""
    from veros.core.operators import at, numpy as npx, update
    from veros.core.surface_forcing import set_surface_freshwater_flux

    state = _make_state(enable_surface_freshwater_flux=True)
    template = state.variables.surface_freshwater_flux

    def t_grid_field(wet_values):
        return update(npx.zeros_like(template), at[2:4, 2], npx.array(wet_values))

    # Cell 1 is rainy and thawing. Cell 2 is dry and freezing. In CESM,
    # precipitation includes rain and snow, runoff is liquid, and ice runoff is
    # non-negative frozen runoff from the runoff router. Sea-ice melt/freezing
    # and its carried salt can have either sign.
    provider_fields = _cesm_like_coupled_provider(
        precipitation=t_grid_field([2.0e-5, 0.2e-5]),
        evaporation=t_grid_field([-0.5e-5, -1.4e-5]),
        melt=t_grid_field([0.3e-5, -0.5e-5]),
        runoff=t_grid_field([0.6e-5, 0.0]),
        ice_runoff=t_grid_field([0.1e-5, 0.0]),
        sea_ice_salt=t_grid_field([1.2e-8, -2.0e-8]),
    )

    surface_mask = update(npx.zeros_like(state.variables.maskT), at[2:4, 2, :], True)
    with state.variables.unlock():
        state.variables.maskT = surface_mask
        state.variables.surface_freshwater_flux = provider_fields["surface_freshwater_flux"]
        state.variables.surface_salt_flux = provider_fields["surface_salt_flux"]

    assert state.var_meta["surface_freshwater_flux"].dims == ("xt", "yt")
    assert state.var_meta["surface_salt_flux"].dims == ("xt", "yt")
    assert state.variables.surface_freshwater_flux.shape == state.variables.maskT[:, :, -1].shape
    onp.testing.assert_allclose(state.variables.surface_freshwater_flux[2:4, 2], [2.5e-5, -1.7e-5])
    onp.testing.assert_allclose(state.variables.surface_salt_flux[2:4, 2], [1.2e-8, -2.0e-8])

    actual = set_surface_freshwater_flux(state).forc_salt_surface

    # Freshwater into the ocean freshens (negative); evaporation and freezing
    # salinify (positive). Direct salt exchange modifies, but does not replace,
    # the virtual-salt-flux contribution.
    onp.testing.assert_allclose(actual[2:4, 2], [-8.5578125e-7, 5.7036875e-7])
    assert onp.count_nonzero(actual) == 2


def test_surface_freshwater_flux_inputs_are_veros_only_for_pyom():
    from veros.pyom_compat import VEROS_TO_PYOM_SETTING, VEROS_TO_PYOM_VAR

    assert VEROS_TO_PYOM_SETTING["rho_freshwater"] is None
    assert VEROS_TO_PYOM_SETTING["surface_salinity_reference"] is None
    assert VEROS_TO_PYOM_SETTING["enable_surface_freshwater_flux"] is None
    assert VEROS_TO_PYOM_VAR["surface_freshwater_flux"] is None
    assert VEROS_TO_PYOM_VAR["surface_salt_flux"] is None


def test_enabled_coupled_flux_replaces_setup_supplied_salinity_flux():
    try:
        from veros.core.surface_forcing import set_surface_freshwater_flux
    except ImportError:
        pytest.fail("surface freshwater flux integration is not implemented")

    from veros.core.operators import numpy as npx

    state = _make_state(enable_surface_freshwater_flux=True)
    with state.variables.unlock():
        state.variables.surface_freshwater_flux = npx.ones_like(state.variables.surface_freshwater_flux)
        state.variables.surface_salt_flux = npx.zeros_like(state.variables.surface_salt_flux)
        state.variables.maskT = npx.ones_like(state.variables.maskT)
        state.variables.forc_salt_surface = npx.full_like(state.variables.forc_salt_surface, 42.0)

    actual = set_surface_freshwater_flux(state).forc_salt_surface

    onp.testing.assert_allclose(actual, -0.0347)


def test_disabled_coupled_flux_preserves_setup_supplied_salinity_flux():
    from veros.core.surface_forcing import set_surface_freshwater_flux
    from veros.core.operators import numpy as npx

    state = _make_state(enable_surface_freshwater_flux=False)
    with state.variables.unlock():
        state.variables.forc_salt_surface = npx.full_like(state.variables.forc_salt_surface, 42.0)

    actual = set_surface_freshwater_flux(state).forc_salt_surface

    onp.testing.assert_allclose(actual, 42.0)
    with pytest.raises(RuntimeError, match="not active"):
        state.variables.surface_freshwater_flux
    with pytest.raises(RuntimeError, match="not active"):
        state.variables.surface_salt_flux


def test_coupled_flux_uses_veros_cyclic_halo_exchange():
    from veros.core.surface_forcing import set_surface_freshwater_flux
    from veros.core.operators import at, numpy as npx, update

    state = _make_state(enable_surface_freshwater_flux=True)
    with state.settings.unlock():
        state.settings.enable_cyclic_x = True

    freshwater_flux = npx.zeros_like(state.variables.surface_freshwater_flux)
    freshwater_flux = update(freshwater_flux, at[2, 2:-2], 1.0)
    freshwater_flux = update(freshwater_flux, at[3, 2:-2], 2.0)

    with state.variables.unlock():
        state.variables.surface_freshwater_flux = freshwater_flux
        state.variables.surface_salt_flux = npx.zeros_like(state.variables.surface_salt_flux)
        state.variables.maskT = npx.ones_like(state.variables.maskT)

    actual = set_surface_freshwater_flux(state).forc_salt_surface

    onp.testing.assert_allclose(actual[:, 2], [-0.0347, -0.0694, -0.0347, -0.0694, -0.0347, -0.0694])


def test_thermodynamics_applies_coupled_flux_before_vertical_mixing(diskless_mode):
    from veros.core import thermodynamics
    from veros.core.operators import numpy as npx
    from veros.setups.acc_basic import ACCBasicSetup

    simulation = ACCBasicSetup(override=dict(enable_surface_freshwater_flux=True))
    simulation.setup()
    state = simulation.state
    initial_surface_salinity = state.variables.salt[:, :, -1, state.variables.tau]

    with state.variables.unlock():
        state.variables.surface_freshwater_flux = 1e-5 * state.variables.maskT[:, :, -1]
        state.variables.surface_salt_flux = npx.zeros_like(state.variables.surface_salt_flux)
        state.variables.forc_salt_surface = npx.zeros_like(state.variables.forc_salt_surface)

    thermodynamics.thermodynamics(state)

    expected_flux = -3.47e-7 * state.variables.maskT[:, :, -1]
    expected_salinity = initial_surface_salinity + state.settings.dt_tracer * expected_flux / state.variables.dzt[-1]
    onp.testing.assert_allclose(state.variables.forc_salt_surface, expected_flux)
    onp.testing.assert_allclose(state.variables.salt[:, :, -1, state.variables.taup1], expected_salinity)


def test_coupled_flux_uses_native_top_cell_tracer_timestep():
    from veros.core import thermodynamics
    from veros.core.operators import at, numpy as npx, update
    from veros.core.surface_forcing import set_surface_freshwater_flux

    state = _make_state(enable_surface_freshwater_flux=True)
    with state.settings.unlock():
        state.settings.dt_tracer = 20.0

    with state.variables.unlock():
        state.variables.dzt = npx.full_like(state.variables.dzt, 10.0)
        state.variables.dzw = npx.full_like(state.variables.dzw, 10.0)
        state.variables.kbot = update(state.variables.kbot, at[2:-2, 2:-2], 1)
        state.variables.maskT = update(state.variables.maskT, at[2:-2, 2:-2, :], True)
        state.variables.surface_freshwater_flux = update(state.variables.surface_freshwater_flux, at[2:-2, 2:-2], 1.0)
        state.variables.salt = update(state.variables.salt, at[2:-2, 2:-2, :, state.variables.taup1], 35.0)
        state.variables.forc_salt_surface = set_surface_freshwater_flux(state).forc_salt_surface

    actual = thermodynamics.vertmix_tempsalt(state).salt[2:-2, 2:-2, -1, state.variables.taup1]

    onp.testing.assert_allclose(actual, 34.9306)
