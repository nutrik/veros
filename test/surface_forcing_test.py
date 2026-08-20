import numpy as onp
import pytest


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

    state = VerosState(VARIABLES, SETTINGS, DIM_TO_SHAPE_VAR)
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
