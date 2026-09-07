// Generated-solver setup, execution, and result extraction.
#include "crane_mpc/ocp_solver.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "crane_ocp/acados_solver.hpp"

extern "C" {
#include "acados_solver_crane_mpc_pzs100.h"  // NOLINT(build/include_subdir)

#include "crane_mpc_pzs100_constraints/crane_mpc_pzs100_constraints.h"
#include "crane_mpc_pzs100_cost/crane_mpc_pzs100_cost.h"
#include "crane_mpc_pzs100_model/crane_mpc_pzs100_model.h"
#include "crane_mpc_pzs100_output.h"  // NOLINT(build/include_subdir)
}

#include "crane_mpc_ocp_generated.h"  // NOLINT(build/include_subdir)

namespace crane_mpc
{
namespace
{

using crane_model::ErrorCode;
using crane_model::Result;
using crane_model::Status;

Status failure(ErrorCode code, std::string message)
{
  return Status{code, std::move(message)};
}

constexpr int kNx = static_cast<int>(kOcpStateDof);
constexpr int kNu = static_cast<int>(kOcpInputDof);
constexpr int kNy = static_cast<int>(Ocp::kStageResidualDof);
constexpr int kNyTerminal = static_cast<int>(Ocp::kTerminalResidualDof);
constexpr int kNh = static_cast<int>(Ocp::kNonlinearConstraintDof);
constexpr int kNsbx = 2 * static_cast<int>(crane_model::kPassiveDof);
constexpr int kNp = CRANE_MPC_OCP_PARAMETER_DOF;
constexpr int kNbx = static_cast<int>(kOcpBoxedStateDof);

static_assert(
  CRANE_MPC_PZS100_NX == kNx,
  "nx is the rigid state plus C3's lagged command and force state; the tool is not planned");
static_assert(
  CRANE_MPC_PZS100_NBX == kNbx && CRANE_MPC_PZS100_NBXN == kNbx,
  "the boxed rows are the rigid state and the lagged command; the force states are held by "
  "constraint 6 and carry no box");
static_assert(CRANE_MPC_PZS100_NBX0 == kNx, "x_0 pins every row, OCP-only states included");
static_assert(
  CRANE_MPC_PZS100_NU == kNu,
  "nu is the five joint commands and the progress acceleration");
static_assert(
  CRANE_MPC_PZS100_NP == kNp, "p is the pinned tool coordinate and the payload body ()");
static_assert(CRANE_MPC_PZS100_NY == kNy, "y = [q_a, dq_a, q_u, dq_u, lag, v_s, tau_a, u]");
static_assert(CRANE_MPC_PZS100_NYN == kNyTerminal, "the terminal cost has no input rows");
static_assert(
  CRANE_MPC_OCP_RESIDUAL_TERMINAL_DOF == kNyTerminal,
  "the terminal residual is the leading prefix of the stage one, so one set of offsets "
  "addresses both");
static_assert(CRANE_MPC_PZS100_NH == kNh, "constraint 6 five times, then constraint 7");
static_assert(CRANE_MPC_PZS100_NHN == 0, "no input at the terminal node, so no tau_a");
static_assert(CRANE_MPC_PZS100_NSBX == kNsbx, "constraints 3 and 4 are the soft box rows");
static_assert(CRANE_MPC_PZS100_NSH == kNh, "hydraulic constraints are soft");
static_assert(CRANE_MPC_PZS100_NBU == kNu, "constraint 5 boxes every planned input");
static_assert(CRANE_MPC_PZS100_N == CRANE_MPC_OCP_HORIZON, "the header and the solver agree");

constexpr std::size_t kReducedActuatedPosition = CRANE_MPC_OCP_STATE_PLANNED_POSITION;
constexpr std::size_t kReducedPassivePosition = CRANE_MPC_OCP_STATE_PASSIVE_POSITION;
constexpr std::size_t kReducedActuatedVelocity = CRANE_MPC_OCP_STATE_PLANNED_VELOCITY;
constexpr std::size_t kReducedPassiveVelocity = CRANE_MPC_OCP_STATE_PASSIVE_VELOCITY;

// C3's two actuator blocks. `kCommandLagAxes` is the export's own list, derived
// there from the fit; the arm is missing from it because its `tau_v` is zero.
constexpr std::size_t kReducedCommandLag = CRANE_MPC_OCP_STATE_COMMAND_LAG;
constexpr std::size_t kReducedProgress = CRANE_MPC_OCP_STATE_PROGRESS;
constexpr std::size_t kReducedProgressRate = CRANE_MPC_OCP_STATE_PROGRESS_RATE;
constexpr std::size_t kReducedActuatedForce = CRANE_MPC_OCP_STATE_ACTUATED_FORCE;
constexpr std::size_t kInputProgressAccel = CRANE_MPC_OCP_INPUT_PROGRESS_ACCEL;

// The axis the lag row of `wiki/mpc.md` 2 is written on, and the rate the
// progress row regulates toward. Both are the export's, and both are decisions
// rather than tuning -- see `scripts/export_ocp.py`.
constexpr std::size_t kLagAxis = CRANE_MPC_OCP_LAG_AXIS;
constexpr double kProgressRateReference = CRANE_MPC_OCP_PROGRESS_RATE_REFERENCE;

static_assert(
  kReducedProgress + 1U == kReducedProgressRate &&
  kReducedProgressRate + 1U == kReducedActuatedForce,
  "the progress pair sits between the lagged command and the force state, so the boxed rows "
  "stay a contiguous prefix and v_s can carry a box at all");
static_assert(
  kInputProgressAccel == kOcpPlannedInputDof,
  "the progress acceleration is the sixth input, after the five joint commands");
constexpr std::array<std::size_t, kOcpCommandLagDof> kCommandLagAxes =
  CRANE_MPC_OCP_STATE_COMMAND_LAG_AXES;

static_assert(
  CRANE_MPC_OCP_STATE_COMMAND_LAG_DOF == static_cast<int>(kOcpCommandLagDof),
  "a refit that gives another axis a command lag changes every offset after it");
static_assert(
  CRANE_MPC_OCP_BOXED_STATE_DOF == static_cast<int>(kOcpBoxedStateDof),
  "the boxed rows are a contiguous prefix of x, so idxbx positions are state rows");

static_assert(
  kReducedPassivePosition == CRANE_MPC_OCP_SOFT_PASSIVE_POSITION,
  "constraint 3 softens the sway rows the export softened");
static_assert(
  kReducedPassiveVelocity == CRANE_MPC_OCP_SOFT_PASSIVE_VELOCITY,
  "constraint 4 softens the sway-rate rows the export softened");

constexpr std::size_t kResidualActuatedPosition = CRANE_MPC_OCP_RESIDUAL_PLANNED_POSITION;
constexpr std::size_t kResidualActuatedVelocity = CRANE_MPC_OCP_RESIDUAL_PLANNED_VELOCITY;
constexpr std::size_t kResidualPassivePosition = CRANE_MPC_OCP_RESIDUAL_PASSIVE_POSITION;
constexpr std::size_t kResidualPassiveVelocity = CRANE_MPC_OCP_RESIDUAL_PASSIVE_VELOCITY;
constexpr std::size_t kResidualLag = CRANE_MPC_OCP_RESIDUAL_LAG;
constexpr std::size_t kResidualProgressRate = CRANE_MPC_OCP_RESIDUAL_PROGRESS_RATE;
constexpr std::size_t kResidualActuatedForce = CRANE_MPC_OCP_RESIDUAL_ACTUATED_FORCE;
constexpr std::size_t kResidualInput = CRANE_MPC_OCP_RESIDUAL_INPUT;

constexpr std::size_t kOutputActuatedForce = 0;
constexpr std::size_t kOutputCylinderForce = crane_model::kActuatedDof;
constexpr std::size_t kOutputPistonVelocity = 2 * crane_model::kActuatedDof;
constexpr std::size_t kOutputAxisFlow = 3 * crane_model::kActuatedDof;
constexpr std::size_t kOutputDof = 4 * crane_model::kActuatedDof;

constexpr std::array<double, Ocp::kNonlinearConstraintDof> kConstraintScale =
  CRANE_MPC_OCP_CONSTRAINT_SCALE;

// The capsule half is `crane_ocp`'s and identical for every generated solver in
// the stack; what is this problem's own is the four CasADi functions the export
// ships beside it.
struct Backend
{
  crane_ocp::AcadosBackend acados{};

  /// acados' implicit residual `f_impl = xdot - f(x, u, p)`. See `explicit_ode`.
  crane_ocp::AcadosFunction implicit_ode{nullptr};
  crane_ocp::AcadosFunction residual{nullptr};
  crane_ocp::AcadosFunction terminal{nullptr};
  crane_ocp::AcadosFunction constraint{nullptr};
  crane_ocp::CasadiFunction output{nullptr};
};

// The one generated solver this package ships, bound to the five CasADi
// functions acados and the export wrote beside it. `constr_h_fun` and the output
// map are pure functions of `(x, u, p)`, so `nonlinear_constraint()` and the
// physical-unit reports read them here rather than off the QP.
Backend pzs100_backend()
{
  Backend backend;
  backend.acados = CRANE_OCP_ACADOS_BACKEND(crane_mpc_pzs100);
  backend.implicit_ode = &crane_mpc_pzs100_impl_dae_fun;
  backend.residual = &crane_mpc_pzs100_cost_y_fun;
  backend.terminal = &crane_mpc_pzs100_cost_y_e_fun;
  backend.constraint = &crane_mpc_pzs100_constr_h_fun;
  backend.output = &crane_mpc_pzs100_output;
  return backend;
}

/// `f(x, u, p)` out of acados' implicit residual, for the callers off the solve path.
/**
 * C3 is stiff at `T_s` and the horizon is integrated with IRK, which generates
 * `f_impl = xdot - f(x, u, p)` and **no** explicit entry point. The explicit
 * right-hand side is therefore that residual at `xdot = 0`, negated. Only
 * `Ocp::dynamics` and `Ocp::propagate` read it; acados integrates the horizon
 * itself and never comes through here.
 */
bool explicit_ode(
  const Backend & backend, const double * x, const double * u, const double * p, double * result)
{
  static const std::array<double, static_cast<std::size_t>(kNx)> rest{};
  if (!crane_ocp::evaluate(
      backend.implicit_ode, {x, rest.data(), u, nullptr, nullptr, p}, result))
  {
    return false;
  }
  for (int row = 0; row < kNx; ++row) {
    result[static_cast<std::size_t>(row)] = -result[static_cast<std::size_t>(row)];
  }
  return true;
}

double pinned_tool(const crane_model::State & x)
{
  return x[static_cast<Eigen::Index>(kStateActuatedPosition + kToolRow)];
}

using ParameterVector = std::array<double, static_cast<std::size_t>(kNp)>;

constexpr std::size_t kParameterToolPosition = CRANE_MPC_OCP_PARAMETER_TOOL_POSITION;
constexpr std::size_t kParameterPayloadMass = CRANE_MPC_OCP_PARAMETER_PAYLOAD_MASS;
constexpr std::size_t kParameterPayloadCom = CRANE_MPC_OCP_PARAMETER_PAYLOAD_COM;
constexpr std::size_t kParameterPayloadInertia = CRANE_MPC_OCP_PARAMETER_PAYLOAD_INERTIA;

// This OCP's own half of `p`: the stage's nominal virtual time and the
// second-order expansion of its reference about it. The cost evaluates the
// reference at the progress state, and a spline cannot be baked into a
// generated solver, so this local model is what carries `q_a,ref(s)` and both
// of its derivatives into the gradient and the Gauss-Newton Hessian.
constexpr std::size_t kParameterProgressNominal = CRANE_MPC_OCP_PARAMETER_PROGRESS_NOMINAL;
constexpr std::size_t kParameterReferencePosition = CRANE_MPC_OCP_PARAMETER_REFERENCE_POSITION;
constexpr std::size_t kParameterReferenceFirst = CRANE_MPC_OCP_PARAMETER_REFERENCE_FIRST;
constexpr std::size_t kParameterReferenceSecond = CRANE_MPC_OCP_PARAMETER_REFERENCE_SECOND;

constexpr std::array<std::array<int, 2>, 6> kInertiaEntries = {
  CRANE_MPC_OCP_PARAMETER_INERTIA_ENTRIES};

/// Write one stage's reference expansion into `p`, in the export's own packing.
void write_reference(ParameterVector & parameter, const OcpStage & stage)
{
  parameter[kParameterProgressNominal] = stage.progress_nominal;
  for (std::size_t row = 0; row < kPlannedDof; ++row) {
    parameter[kParameterReferencePosition + row] = stage.q_a_ref[row];
    parameter[kParameterReferenceFirst + row] = stage.dq_a_ref[row];
    parameter[kParameterReferenceSecond + row] = stage.ddq_a_ref[row];
  }
}

ParameterVector parameter_vector(
  const crane_model::State & x, const crane_model::Payload & payload)
{
  ParameterVector parameter{};
  parameter[kParameterToolPosition] = pinned_tool(x);
  parameter[kParameterPayloadMass] = payload.mass_kg;
  for (std::size_t row = 0; row < 3U; ++row) {
    parameter[kParameterPayloadCom + row] =
      payload.center_of_mass_k8_m[static_cast<Eigen::Index>(row)];
  }
  for (std::size_t entry = 0; entry < kInertiaEntries.size(); ++entry) {
    parameter[kParameterPayloadInertia + entry] = payload.inertia_k8_kg_m2(
      static_cast<Eigen::Index>(kInertiaEntries[entry][0]),
      static_cast<Eigen::Index>(kInertiaEntries[entry][1]));
  }
  return parameter;
}

bool same_payload(const crane_model::Payload & one, const crane_model::Payload & other)
{
  return one.valid == other.valid && one.mass_kg == other.mass_kg &&
         one.center_of_mass_k8_m == other.center_of_mass_k8_m &&
         one.inertia_k8_kg_m2 == other.inertia_k8_kg_m2;
}

Status check_payload(const crane_model::Payload & payload)
{
  if (!payload.valid) {
    return failure(
      ErrorCode::InvalidArgument,
      "the payload must be declared: crane_model refuses valid == false outright (contract 4), and "
      "a declared empty payload -- valid, zero mass, zero inertia -- is what an empty gripper is. "
      "'Nobody has said' is not a load");
  }
  if (!std::isfinite(payload.mass_kg) || payload.mass_kg < 0.0) {
    return failure(
      ErrorCode::InvalidArgument,
      "the payload mass is " + std::to_string(payload.mass_kg) +
      " kg, and a mass must be finite and >= 0: a negative one is not a light load, and a "
      "non-finite one reaches every stage of the horizon at once through p");
  }
  if (!payload.center_of_mass_k8_m.allFinite() || !payload.inertia_k8_kg_m2.allFinite()) {
    return failure(
      ErrorCode::NonFiniteInput,
      "the payload's centre of mass and inertia must be finite; they are the moment arm and the "
      "second moment the dynamics carry, and a non-finite entry is a solve that answers NaN");
  }
  return Status{};
}

// I_a and I_u in canonical indices, the same rows `crane_model`'s `kActuatedRows`
// and `kPassiveRows` name. Needed here because `crane_model::State` orders the
// coordinates actuated-then-passive and `crane_model::Q` orders them canonically.
constexpr std::array<std::size_t, crane_model::kActuatedDof> kActuatedCanonical{
  {0, 1, 2, 3, 6, 7}};
constexpr std::array<std::size_t, crane_model::kPassiveDof> kPassiveCanonical{{4, 5}};

/// Where C3's two actuator states start when nothing has carried them forward.
/**
 * There is no force measurement anywhere in the stack, so the force state has to
 * be seeded from the model: `h_eff` of `wiki/robot_model.md` §3.4 is the force
 * that holds the machine still at this configuration, which is what the machine
 * is doing when a horizon is first posed. Seeding it at zero would start every
 * prediction with the hydraulics switched off and the boom in free fall. The
 * lagged command is seeded at the measured velocity, the PT1's own steady state.
 */
/// `h_eff` at `x`: the actuated force that holds the machine still there.
/**
 * `robot_model.md` 3.4. Two consumers, and they want the same number for the
 * same reason -- there is no force measurement anywhere in the stack, so the
 * model is what says what holding still costs: it seeds C3's force state, and
 * it is the **reference** the effort term of `wiki/mpc.md` 2 prices `tau_a`
 * against. `fallback` is returned unchanged where the model refuses.
 */
std::array<double, kPlannedDof> static_hold_force(
  const crane_model::Model & model, const crane_model::State & x,
  const crane_model::Payload & payload, std::array<double, kPlannedDof> fallback)
{
  crane_model::Q q = crane_model::Q::Zero();
  crane_model::DQ dq = crane_model::DQ::Zero();
  for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
    q[static_cast<Eigen::Index>(kActuatedCanonical[row])] =
      x[static_cast<Eigen::Index>(kStateActuatedPosition + row)];
    dq[static_cast<Eigen::Index>(kActuatedCanonical[row])] =
      x[static_cast<Eigen::Index>(kStateActuatedVelocity + row)];
  }
  for (std::size_t row = 0; row < crane_model::kPassiveDof; ++row) {
    q[static_cast<Eigen::Index>(kPassiveCanonical[row])] =
      x[static_cast<Eigen::Index>(kStatePassivePosition + row)];
    dq[static_cast<Eigen::Index>(kPassiveCanonical[row])] =
      x[static_cast<Eigen::Index>(kStatePassiveVelocity + row)];
  }
  const auto reduced =
    model.reduced_actuated_dynamics(q, dq, crane_model::Input::Zero(), payload);
  if (reduced.ok()) {
    for (std::size_t row = 0; row < kPlannedDof; ++row) {
      fallback[row] = reduced.value().bias_eff[static_cast<Eigen::Index>(row)];
    }
  }
  return fallback;
}

OcpOnlyState seed_ocp_only(
  const crane_model::Model & model, const crane_model::State & x,
  const crane_model::Payload & payload, OcpOnlyState seeded)
{
  for (std::size_t row = 0; row < kPlannedDof; ++row) {
    seeded.command_lag[row] = x[static_cast<Eigen::Index>(kStateActuatedVelocity + row)];
  }
  seeded.force = static_hold_force(model, x, payload, seeded.force);
  // The plan is spent at nominal rate until a solve says otherwise: seeding the
  // rate anywhere else would start every first horizon already behind or ahead
  // of a plan nobody has yet decided to slow down.
  seeded.progress = 0.0;
  seeded.progress_rate = kProgressRateReference;
  return seeded;
}

/// The canonical state, the actuator state, and the OCP vector that holds both.
/**
 * Three categories, not two: `crane_model::State` is the model API contract's
 * fixed sixteen and has no slot for `u_f` or `tau_a`, so those arrive here
 * separately rather than by widening a contract the planner and the collision
 * model also read.
 */
std::vector<double> reduce(const crane_model::State & x, const OcpOnlyState & ocp_only)
{
  std::vector<double> rows(static_cast<std::size_t>(kNx), 0.0);
  for (std::size_t slot = 0; slot < kOcpCommandLagDof; ++slot) {
    rows[kReducedCommandLag + slot] = ocp_only.command_lag[kCommandLagAxes[slot]];
  }
  rows[kReducedProgress] = ocp_only.progress;
  rows[kReducedProgressRate] = ocp_only.progress_rate;
  for (std::size_t row = 0; row < kPlannedDof; ++row) {
    rows[kReducedActuatedForce + row] = ocp_only.force[row];
  }
  for (std::size_t row = 0; row < kPlannedDof; ++row) {
    rows[kReducedActuatedPosition + row] =
      x[static_cast<Eigen::Index>(kStateActuatedPosition + row)];
    rows[kReducedActuatedVelocity + row] =
      x[static_cast<Eigen::Index>(kStateActuatedVelocity + row)];
  }
  for (std::size_t row = 0; row < crane_model::kPassiveDof; ++row) {
    rows[kReducedPassivePosition + row] =
      x[static_cast<Eigen::Index>(kStatePassivePosition + row)];
    rows[kReducedPassiveVelocity + row] =
      x[static_cast<Eigen::Index>(kStatePassiveVelocity + row)];
  }
  return rows;
}

/// The OCP-only half of an OCP vector, for the axes that have no lag state `u`.
OcpOnlyState expand_ocp_only(const std::vector<double> & rows, const std::vector<double> & u)
{
  OcpOnlyState ocp_only;
  for (std::size_t row = 0; row < kPlannedDof; ++row) {
    // No lag state means the pole is at infinity, so the lagged command is the
    // command. Overwritten below for the axes that do carry one.
    ocp_only.command_lag[row] = u.empty() ? 0.0 : u[row];
    ocp_only.force[row] = rows[kReducedActuatedForce + row];
  }
  for (std::size_t slot = 0; slot < kOcpCommandLagDof; ++slot) {
    ocp_only.command_lag[kCommandLagAxes[slot]] = rows[kReducedCommandLag + slot];
  }
  ocp_only.progress = rows[kReducedProgress];
  ocp_only.progress_rate = rows[kReducedProgressRate];
  return ocp_only;
}

crane_model::State expand(const std::vector<double> & rows, double q_tool)
{
  crane_model::State x = crane_model::State::Zero();
  for (std::size_t row = 0; row < kPlannedDof; ++row) {
    x[static_cast<Eigen::Index>(kStateActuatedPosition + row)] =
      rows[kReducedActuatedPosition + row];
    x[static_cast<Eigen::Index>(kStateActuatedVelocity + row)] =
      rows[kReducedActuatedVelocity + row];
  }
  for (std::size_t row = 0; row < crane_model::kPassiveDof; ++row) {
    x[static_cast<Eigen::Index>(kStatePassivePosition + row)] =
      rows[kReducedPassivePosition + row];
    x[static_cast<Eigen::Index>(kStatePassiveVelocity + row)] =
      rows[kReducedPassiveVelocity + row];
  }
  x[static_cast<Eigen::Index>(kStateActuatedPosition + kToolRow)] = q_tool;
  x[static_cast<Eigen::Index>(kStateActuatedVelocity + kToolRow)] = 0.0;
  return x;
}

std::vector<double> reduce_input(const crane_model::Input & u, double progress_accel)
{
  std::vector<double> rows(static_cast<std::size_t>(kNu), 0.0);
  for (std::size_t row = 0; row < kPlannedDof; ++row) {
    rows[row] = u[static_cast<Eigen::Index>(row)];
  }
  // The progress acceleration is not an actuated axis, so it has no slot in
  // `crane_model::Input`. Every caller off the solve path passes zero: it moves
  // no joint and reaches neither the output map nor the constraint rows.
  rows[kInputProgressAccel] = progress_accel;
  return rows;
}

crane_model::Input expand_input(const std::vector<double> & rows)
{
  crane_model::Input u = crane_model::Input::Zero();
  for (std::size_t row = 0; row < kPlannedDof; ++row) {
    u[static_cast<Eigen::Index>(row)] = rows[row];
  }
  return u;
}

bool all_finite(const std::array<double, crane_model::kActuatedDof> & values)
{
  return std::all_of(
    values.begin(), values.end(), [](double value) {return std::isfinite(value);});
}

bool all_finite(const std::array<double, crane_model::kPassiveDof> & values)
{
  return std::all_of(
    values.begin(), values.end(), [](double value) {return std::isfinite(value);});
}

bool all_non_negative(const std::array<double, crane_model::kActuatedDof> & values)
{
  return std::all_of(
    values.begin(), values.end(),
    [](double value) {return std::isfinite(value) && value >= 0.0;});
}

bool all_non_negative(const std::array<double, crane_model::kPassiveDof> & values)
{
  return std::all_of(
    values.begin(), values.end(),
    [](double value) {return std::isfinite(value) && value >= 0.0;});
}

bool all_positive(const std::array<double, crane_model::kActuatedDof> & values)
{
  return std::all_of(
    values.begin(), values.end(),
    [](double value) {return std::isfinite(value) && value > 0.0;});
}

bool all_positive(const std::array<double, crane_model::kPassiveDof> & values)
{
  return std::all_of(
    values.begin(), values.end(),
    [](double value) {return std::isfinite(value) && value > 0.0;});
}

Status check_settings(const OcpSettings & settings)
{
  if (!std::isfinite(settings.sample_time_s) || settings.sample_time_s <= 0.0) {
    return failure(
      ErrorCode::InvalidArgument,
      "T_s must be finite and positive; it is the control cycle of mpc 4, the knot spacing of the "
      "horizon and the ERK4 step, and those are one number rather than three");
  }
  if (settings.horizon_length < 2U) {
    return failure(
      ErrorCode::InvalidArgument,
      "N must be at least two shooting intervals; mpc 4 sizes it by the sway period and a horizon "
      "shorter than one period leaves the sway terms nearly free to violate");
  }
  if (!std::isfinite(settings.solve_budget_s) || settings.solve_budget_s <= 0.0) {
    return failure(ErrorCode::InvalidArgument, "the solve budget must be finite and positive");
  }
  if (!std::isfinite(settings.levenberg_marquardt) || settings.levenberg_marquardt < 0.0) {
    return failure(
      ErrorCode::InvalidArgument, "the Levenberg-Marquardt regularisation must be finite and >= 0");
  }
  const CostWeights & weights = settings.weights;
  if (!all_non_negative(weights.q_a) || !all_non_negative(weights.dq_a) ||
    !all_non_negative(weights.q_u) || !all_non_negative(weights.dq_u) ||
    !all_non_negative(weights.tau_a) || !all_non_negative(weights.u))
  {
    return failure(
      ErrorCode::InvalidArgument,
      "every weight of mpc 2 must be finite and non-negative: the Gauss-Newton Hessian is J' W J "
      "and a negative entry is what turns 'positive semi-definite by construction' into a hope");
  }
  if (!std::isfinite(weights.terminal_scale) || weights.terminal_scale < 0.0) {
    return failure(ErrorCode::InvalidArgument, "the terminal weight scale must be finite and >= 0");
  }
  if (!std::isfinite(weights.lag) || weights.lag < 0.0 ||
    !std::isfinite(weights.progress_accel) || weights.progress_accel < 0.0)
  {
    return failure(
      ErrorCode::InvalidArgument,
      "the lag and progress-acceleration weights must be finite and >= 0, for the same reason "
      "every other weight must: the Gauss-Newton Hessian is J' W J");
  }
  if (!std::isfinite(weights.progress_rate) || weights.progress_rate <= 0.0) {
    return failure(
      ErrorCode::InvalidArgument,
      "the progress-rate weight must be finite and **positive**: it is the only price on spending "
      "time, and at zero the optimizer stops the plan for free wherever tracking is hard, which "
      "is a controller that never arrives rather than one that tracks time-indexed");
  }
  const BoxLimits & limits = settings.limits;
  if (!all_finite(limits.q_a_lower) || !all_finite(limits.q_a_upper)) {
    return failure(
      ErrorCode::InvalidArgument,
      "constraint 1 of mpc 3 needs a finite control-safe range on every actuated row "
      "(parameters.md 2 -- not the URDF)");
  }
  for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
    if (!(limits.q_a_lower[row] < limits.q_a_upper[row])) {
      return failure(
        ErrorCode::InvalidArgument,
        "actuated row " + std::to_string(row) + " has an empty control-safe position range");
    }
  }
  if (!all_positive(limits.dq_a_max) || !all_positive(limits.q_u_max) ||
    !all_positive(limits.dq_u_max) || !all_positive(limits.u_max))
  {
    return failure(
      ErrorCode::InvalidArgument,
      "constraints 2 to 5 of mpc 3 each need a finite positive bound; an invented or absent one is "
      "the silent stub the model API contract 5 exists to prevent");
  }
  if (!std::isfinite(limits.progress_rate_max) ||
    limits.progress_rate_max < kProgressRateReference)
  {
    return failure(
      ErrorCode::InvalidArgument,
      "the progress-rate ceiling must be finite and at least one; below one the reference could "
      "never be spent at its own nominal rate, which is the behaviour every other page in the "
      "wiki describes");
  }
  if (!std::isfinite(limits.progress_accel_max) || limits.progress_accel_max <= 0.0) {
    return failure(
      ErrorCode::InvalidArgument,
      "the progress-acceleration bound must be finite and positive; at zero the progress rate is "
      "frozen at whatever it was carried in with and the state is decoration");
  }
  if (!all_non_negative(limits.q_a_margin)) {
    return failure(
      ErrorCode::InvalidArgument,
      "constraint 1's safety margin must be finite and >= 0 on every row; zero is legal and means "
      "the bound is the control-safe limit itself");
  }
  for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
    if (!(limits.q_a_lower[row] + limits.q_a_margin[row] <
      limits.q_a_upper[row] - limits.q_a_margin[row]))
    {
      return failure(
        ErrorCode::InvalidArgument,
        "actuated row " + std::to_string(row) +
        "'s safety margin closes its own control-safe range; a margin is a tightening, not a "
        "replacement for the limit");
    }
  }
  const HydraulicLimits & hydraulics = settings.hydraulics;
  if (!std::isfinite(hydraulics.pump_flow_max) || hydraulics.pump_flow_max <= 0.0) {
    return failure(
      ErrorCode::InvalidArgument,
      "constraint 7 of mpc 3 needs a finite positive Q_P^max; parameters.md 4 states it once, at "
      "1.4e-3 m^3/s measured 2023 pre-retrofit and not re-verified, and a weak number with its "
      "provenance recorded is worth more than an absent constraint");
  }
  if (!std::isfinite(hydraulics.pump_flow_planning_factor) ||
    hydraulics.pump_flow_planning_factor <= 0.0 || hydraulics.pump_flow_planning_factor > 1.0)
  {
    return failure(
      ErrorCode::InvalidArgument,
      "the pump flow planning factor is parameters.md 4's 0.95x discount on Q_P^max and must lie "
      "in (0, 1]; it is not a second kappa and it is not a place to buy margin back");
  }
  if (!std::isfinite(hydraulics.system_pressure_pa) || hydraulics.system_pressure_pa <= 0.0) {
    return failure(
      ErrorCode::InvalidArgument,
      "constraint 6 of mpc 3 needs a finite positive relief pressure to derive F_i^max from "
      "(robot_model 4.6); it is the one number that cannot come off the description and "
      "parameters.md 7 lists the pressure constants among its gaps");
  }
  const SlackWeights & slack = settings.slack;
  if (!all_positive(slack.q_u) || !all_positive(slack.dq_u) ||
    !all_positive(slack.cylinder_force) || !std::isfinite(slack.pump_flow) ||
    slack.pump_flow <= 0.0)
  {
    return failure(
      ErrorCode::InvalidArgument,
      "every L1 slack price of mpc 3.2 must be finite and positive; a soft constraint priced at "
      "zero is not a soft constraint, it is an absent one, and it would be absent silently");
  }
  return Status{};
}

}  // namespace

struct Ocp::Impl
{
  OcpSettings settings{};
  Backend backend{};

    /// The shipped solver, opened and freed by `crane_ocp`.
  std::unique_ptr<crane_ocp::AcadosSolver> acados;

    crane_model::Payload payload{};
  bool payload_changed{false};

    /// Kept for `h_eff`: C3's force-state seed and the effort term's reference.
  std::optional<crane_model::Model> model;

    /// The OCP-only state at `x_0`, carried from the last solve.
  OcpOnlyState ocp_only{};
  bool ocp_only_seeded{false};

    /// `h_eff` at `x_0`, the effort term's reference. See `Ocp::solve`.
  std::array<double, kPlannedDof> force_reference{};

  std::array<CylinderForceLimit, crane_model::kActuatedDof> cylinder_force_max{};
  double flow_max{};
  std::vector<double> constraint_scale{};

  std::vector<double> weight_diagonal{};
  std::vector<double> terminal_weight_diagonal{};
  std::vector<double> weight_matrix{};
  std::vector<double> terminal_weight_matrix{};

  std::vector<double> lower_constraint{};
  std::vector<double> upper_constraint{};
  std::vector<double> input_lower{};
  std::vector<double> input_upper{};

  std::vector<double> reference{};
  std::vector<double> terminal_reference{};
  std::vector<double> lower_state{};
  std::vector<double> upper_state{};
  std::vector<double> guess{};
  std::vector<double> slack_rest{};

  std::vector<double> slack_price_path{};
  std::vector<double> slack_price_initial{};
  std::vector<double> slack_price_terminal{};

  std::vector<int> slack_count{};

  Impl() = default;
  ~Impl() = default;
  Impl(const Impl &) = delete;
  Impl & operator=(const Impl &) = delete;
  Impl(Impl &&) = delete;
  Impl & operator=(Impl &&) = delete;

  [[nodiscard]] int intervals() const noexcept
  {
    return static_cast<int>(settings.horizon_length);
  }
};

// `position_box`'s three properties, at compile time. There is no test target in
// this package -- issue 104 stripped it to the offline harness deliberately --
// and these are the assertions that harness cannot make, because a run only ever
// exercises the margin it was configured with.
static_assert(
  position_box(-1.2, 1.563, 0.0, 0.0) == std::pair<double, double>{-1.2, 1.563},
  "a zero margin has to reproduce the untightened bound exactly, or the margin "
  "can drift without anything noticing");
static_assert(
  position_box(-1.0, 1.5, 0.25, 0.0) == std::pair<double, double>{-0.75, 1.25},
  "a margin tightens both ends of constraint 1's box");
static_assert(
  position_box(0.0, 2.0, 0.25, 0.0) == std::pair<double, double>{0.0, 1.75},
  "a margin may not exclude the pose the machine is in: a fully retracted "
  "telescope sits exactly on q_a_lower[3] = 0 and the problem stays feasible "
  "there. The numbers are binary-exact so the assertion is about the rule and "
  "not about rounding");

double ConstraintSlack::worst() const noexcept
{
  double largest = pump_flow;
  for (const double value : q_u) {largest = std::max(largest, value);}
  for (const double value : dq_u) {largest = std::max(largest, value);}
  for (const double value : cylinder_force) {largest = std::max(largest, value);}
  return largest;
}

double ConstraintViolation::worst() const noexcept
{
  return std::max(std::max(q_u, dq_u), std::max(cylinder_force, pump_flow));
}

double CostTerms::total() const noexcept
{
  return q_a + dq_a + q_u + dq_u + lag + progress + tau_a + u + terminal + slack;
}

const char * to_string(SolveOutcome outcome) noexcept
{
  switch (outcome) {
    case SolveOutcome::Converged: return "converged";
    case SolveOutcome::BudgetExceeded: return "the solve budget was exceeded";
    case SolveOutcome::Failed: return "the solve failed";
  }
  return "unknown";
}

Ocp::Ocp(std::unique_ptr<Impl> impl) noexcept
: impl_(std::move(impl)) {}

Ocp::~Ocp() = default;

const OcpSettings & Ocp::settings() const noexcept
{
  return impl_->settings;
}

void Ocp::set_solve_budget(double seconds) noexcept
{
  if (std::isfinite(seconds) && seconds > 0.0) {
    impl_->settings.solve_budget_s = seconds;
  }
}

const crane_model::Payload & Ocp::payload() const noexcept
{
  return impl_->payload;
}

Status Ocp::set_payload(const crane_model::Payload & payload)
{
  const Status status = check_payload(payload);
  if (!status.ok()) {
    return status;
  }
  if (same_payload(payload, impl_->payload)) {
    return Status{};
  }
  impl_->payload = payload;
  impl_->payload_changed = true;
  return Status{};
}

Result<std::unique_ptr<Ocp>> Ocp::create(
  const crane_model::ModelConfig & config, const crane_model::Payload & payload,
  const OcpSettings & settings)
{
  using Handle = std::unique_ptr<Ocp>;

  const Status settings_status = check_settings(settings);
  if (!settings_status.ok()) {
    return Result<Handle>::failure(settings_status);
  }

  const Status payload_status = check_payload(payload);
  if (!payload_status.ok()) {
    return Result<Handle>::failure(payload_status);
  }

  auto model = crane_model::Model::create(config);
  if (!model.ok()) {
    return Result<Handle>::failure(model.status());
  }
  auto force_limits = crane_model::derive_cylinder_force_limits(
    model.value(), settings.hydraulics.system_pressure_pa);
  if (!force_limits.ok()) {
    return Result<Handle>::failure(force_limits.status());
  }

  auto impl = std::make_unique<Impl>();
  impl->settings = settings;
  impl->payload = payload;
  impl->model.emplace(std::move(model).value());
  impl->backend = pzs100_backend();
  impl->cylinder_force_max = force_limits.value();
  impl->flow_max =
    settings.hydraulics.pump_flow_planning_factor * settings.hydraulics.pump_flow_max;

  impl->constraint_scale.assign(kConstraintScale.begin(), kConstraintScale.end());

  impl->acados = std::make_unique<crane_ocp::AcadosSolver>(impl->backend.acados);
  const int created = impl->acados->open(
    static_cast<int>(settings.horizon_length), settings.sample_time_s);
  if (created == crane_ocp::AcadosSolver::kCapsuleUnavailable) {
    return Result<Handle>::failure(
      failure(
        ErrorCode::BackendUnavailable,
        std::string("the generated solver ") + impl->acados->name() +
        " would not allocate its capsule"));
  }
  if (created != ACADOS_SUCCESS) {
    return Result<Handle>::failure(
      failure(
        ErrorCode::BackendUnavailable,
        std::string("the generated solver ") + impl->acados->name() +
        " would not create: " + crane_ocp::status_word(created)));
  }

  impl->acados->set_option(
    "levenberg_marquardt", const_cast<double *>(&settings.levenberg_marquardt));

  const int intervals = impl->intervals();


  impl->weight_diagonal.assign(static_cast<std::size_t>(kNy), 0.0);
  impl->terminal_weight_diagonal.assign(static_cast<std::size_t>(kNyTerminal), 0.0);
  const CostWeights & weights = settings.weights;
  for (std::size_t row = 0; row < kPlannedDof; ++row) {
    impl->weight_diagonal[kResidualActuatedPosition + row] = weights.q_a[row];
    impl->weight_diagonal[kResidualActuatedVelocity + row] = weights.dq_a[row];
    impl->weight_diagonal[kResidualActuatedForce + row] = weights.tau_a[row];
    impl->weight_diagonal[kResidualInput + row] = weights.u[row];
    impl->terminal_weight_diagonal[kResidualActuatedPosition + row] =
      weights.terminal_scale * weights.q_a[row];
    impl->terminal_weight_diagonal[kResidualActuatedVelocity + row] =
      weights.terminal_scale * weights.dq_a[row];
  }
  for (std::size_t row = 0; row < crane_model::kPassiveDof; ++row) {
    impl->weight_diagonal[kResidualPassivePosition + row] = weights.q_u[row];
    impl->weight_diagonal[kResidualPassiveVelocity + row] = weights.dq_u[row];
    impl->terminal_weight_diagonal[kResidualPassivePosition + row] =
      weights.terminal_scale * weights.q_u[row];
    impl->terminal_weight_diagonal[kResidualPassiveVelocity + row] =
      weights.terminal_scale * weights.dq_u[row];
  }
  // The lag and progress rows are inside the terminal prefix, so they are priced
  // at the terminal node too: a horizon that ends behind, or ends spending the
  // plan at the wrong rate, pays for that beyond the horizon and not inside it.
  impl->weight_diagonal[kResidualLag] = weights.lag;
  impl->weight_diagonal[kResidualProgressRate] = weights.progress_rate;
  impl->weight_diagonal[kResidualInput + kInputProgressAccel] = weights.progress_accel;
  impl->terminal_weight_diagonal[kResidualLag] = weights.terminal_scale * weights.lag;
  impl->terminal_weight_diagonal[kResidualProgressRate] =
    weights.terminal_scale * weights.progress_rate;
  impl->weight_matrix.assign(static_cast<std::size_t>(kNy) * static_cast<std::size_t>(kNy), 0.0);
  for (std::size_t row = 0; row < impl->weight_diagonal.size(); ++row) {
    impl->weight_matrix[row * static_cast<std::size_t>(kNy) + row] = impl->weight_diagonal[row];
  }
  impl->terminal_weight_matrix.assign(
    static_cast<std::size_t>(kNyTerminal) * static_cast<std::size_t>(kNyTerminal), 0.0);
  for (std::size_t row = 0; row < impl->terminal_weight_diagonal.size(); ++row) {
    impl->terminal_weight_matrix[row * static_cast<std::size_t>(kNyTerminal) + row] =
      impl->terminal_weight_diagonal[row];
  }

  impl->input_lower.resize(static_cast<std::size_t>(kNu));
  impl->input_upper.resize(static_cast<std::size_t>(kNu));
  for (std::size_t index = 0; index < kOcpPlannedInputDof; ++index) {
    impl->input_lower[index] = -settings.limits.u_max[index];
    impl->input_upper[index] = settings.limits.u_max[index];
  }
  impl->input_lower[kInputProgressAccel] = -settings.limits.progress_accel_max;
  impl->input_upper[kInputProgressAccel] = settings.limits.progress_accel_max;
  impl->reference.assign(static_cast<std::size_t>(kNy), 0.0);
  impl->terminal_reference.assign(static_cast<std::size_t>(kNyTerminal), 0.0);
  impl->lower_state.assign(static_cast<std::size_t>(kNx), 0.0);
  impl->upper_state.assign(static_cast<std::size_t>(kNx), 0.0);
  impl->guess.assign(static_cast<std::size_t>(kNx), 0.0);

  impl->lower_constraint.assign(static_cast<std::size_t>(kNh), -1.0);
  impl->upper_constraint.assign(static_cast<std::size_t>(kNh), 1.0);
  for (std::size_t row = 0; row < kPlannedDof; ++row) {
    const double scale = impl->constraint_scale[Ocp::kConstraintCylinderForce + row];
    impl->lower_constraint[Ocp::kConstraintCylinderForce + row] =
      -impl->cylinder_force_max[row].retract / scale;
    impl->upper_constraint[Ocp::kConstraintCylinderForce + row] =
      impl->cylinder_force_max[row].extend / scale;
  }
  impl->lower_constraint[Ocp::kConstraintPumpFlow] = 0.0;
  impl->upper_constraint[Ocp::kConstraintPumpFlow] =
    impl->flow_max / impl->constraint_scale[Ocp::kConstraintPumpFlow];

  impl->slack_rest.assign(static_cast<std::size_t>(kNsbx + kNh), 0.0);

  const SlackWeights & slack = settings.slack;
  std::vector<double> soft_state_price;
  soft_state_price.reserve(static_cast<std::size_t>(kNsbx));
  for (std::size_t row = 0; row < crane_model::kPassiveDof; ++row) {
    soft_state_price.push_back(slack.q_u[row]);
  }
  for (std::size_t row = 0; row < crane_model::kPassiveDof; ++row) {
    soft_state_price.push_back(slack.dq_u[row]);
  }
  std::vector<double> soft_nonlinear_price;
  soft_nonlinear_price.reserve(static_cast<std::size_t>(kNh));
  for (std::size_t row = 0; row < kPlannedDof; ++row) {
    soft_nonlinear_price.push_back(slack.cylinder_force[row]);
  }
  soft_nonlinear_price.push_back(slack.pump_flow);

  impl->slack_price_initial = soft_nonlinear_price;
  impl->slack_price_terminal = soft_state_price;
  impl->slack_price_path = soft_state_price;
  impl->slack_price_path.insert(
    impl->slack_price_path.end(), soft_nonlinear_price.begin(), soft_nonlinear_price.end());

  impl->slack_count.assign(static_cast<std::size_t>(intervals) + 1U, kNsbx + kNh);
  impl->slack_count.front() = kNh;
  impl->slack_count.back() = kNsbx;

  for (int stage = 0; stage <= intervals; ++stage) {
    std::vector<double> & price = stage == 0 ?
      impl->slack_price_initial :
      (stage < intervals ? impl->slack_price_path : impl->slack_price_terminal);
    if (!price.empty()) {
      impl->acados->set_cost(stage, "Zl", impl->slack_rest.data());
      impl->acados->set_cost(stage, "Zu", impl->slack_rest.data());
      impl->acados->set_cost(stage, "zl", price.data());
      impl->acados->set_cost(stage, "zu", price.data());
    }

    if (stage < intervals) {
      impl->acados->set_cost(stage, "W", impl->weight_matrix.data());
      impl->acados->set_constraint(stage, "lh", impl->lower_constraint.data());
      impl->acados->set_constraint(stage, "uh", impl->upper_constraint.data());
      impl->acados->set_constraint(stage, "lbu", impl->input_lower.data());
      impl->acados->set_constraint(stage, "ubu", impl->input_upper.data());
    } else {
      impl->acados->set_cost(stage, "W", impl->terminal_weight_matrix.data());
    }
  }

  return Result<Handle>::success(Handle(new Ocp(std::move(impl))));
}

Result<crane_model::State> Ocp::dynamics(
  const crane_model::State & x, const crane_model::Input & u) const
{
  if (!x.allFinite() || !u.allFinite()) {
    return Result<crane_model::State>::failure(
      failure(ErrorCode::NonFiniteInput, "the state and the input must both be finite"));
  }
  const std::vector<double> state = reduce(x, impl_->ocp_only);
  const std::vector<double> input = reduce_input(u, 0.0);
  const ParameterVector parameter = parameter_vector(x, impl_->payload);
  std::vector<double> rows(static_cast<std::size_t>(kNx), 0.0);
  if (!explicit_ode(
      impl_->backend, state.data(), input.data(), parameter.data(), rows.data()))
  {
    return Result<crane_model::State>::failure(
      failure(
        ErrorCode::SymbolicBackendFailure,
        "the dynamics acados integrates could not be evaluated"));
  }
  const crane_model::State xdot = expand(rows, 0.0);
  if (!xdot.allFinite()) {
    return Result<crane_model::State>::failure(
      failure(ErrorCode::SymbolicBackendFailure, "the dynamics answered a non-finite state rate"));
  }
  return Result<crane_model::State>::success(xdot);
}

Result<crane_model::Vector6> Ocp::actuated_force(
  const crane_model::State & x, const crane_model::Input & u) const
{
  return output_block(x, u, kOutputActuatedForce);
}

Result<crane_model::Vector6> Ocp::cylinder_force(
  const crane_model::State & x, const crane_model::Input & u) const
{
  return output_block(x, u, kOutputCylinderForce);
}

Result<crane_model::Vector6> Ocp::axis_flow(
  const crane_model::State & x, const crane_model::Input & u) const
{
  return output_block(x, u, kOutputAxisFlow);
}

Result<crane_model::Vector6> Ocp::output_block(
  const crane_model::State & x, const crane_model::Input & u, std::size_t offset) const
{
  if (!x.allFinite() || !u.allFinite()) {
    return Result<crane_model::Vector6>::failure(
      failure(ErrorCode::NonFiniteInput, "the state and the input must both be finite"));
  }
  const std::vector<double> state = reduce(x, impl_->ocp_only);
  const std::vector<double> input = reduce_input(u, 0.0);
  const ParameterVector parameter = parameter_vector(x, impl_->payload);
  std::vector<double> rows(kOutputDof, 0.0);
  if (!crane_ocp::evaluate(
      impl_->backend.output, {state.data(), input.data(), parameter.data()}, rows.data()))
  {
    return Result<crane_model::Vector6>::failure(
      failure(
        ErrorCode::SymbolicBackendFailure,
        "the output map of mpc 3's constraints 6 and 7 could not be evaluated"));
  }
  crane_model::Vector6 block;
  for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
    block[static_cast<Eigen::Index>(row)] = rows.at(offset + row);
  }
  return Result<crane_model::Vector6>::success(block);
}

Result<std::vector<double>> Ocp::nonlinear_constraint(
  const crane_model::State & x, const crane_model::Input & u) const
{
  if (!x.allFinite() || !u.allFinite()) {
    return Result<std::vector<double>>::failure(
      failure(ErrorCode::NonFiniteInput, "the state and the input must both be finite"));
  }
  const std::vector<double> state = reduce(x, impl_->ocp_only);
  const std::vector<double> input = reduce_input(u, 0.0);
  const ParameterVector parameter = parameter_vector(x, impl_->payload);
  std::vector<double> rows(static_cast<std::size_t>(kNh), 0.0);
  if (!crane_ocp::evaluate(
      impl_->backend.constraint, {state.data(), input.data(), nullptr, parameter.data()},
      rows.data()))
  {
    return Result<std::vector<double>>::failure(
      failure(
        ErrorCode::SymbolicBackendFailure, "constraints 6 and 7 of mpc 3 could not be evaluated"));
  }
  return Result<std::vector<double>>::success(std::move(rows));
}

const std::vector<double> & Ocp::constraint_lower() const noexcept
{
  return impl_->lower_constraint;
}

const std::vector<double> & Ocp::constraint_upper() const noexcept
{
  return impl_->upper_constraint;
}

const std::vector<double> & Ocp::constraint_scale() const noexcept
{
  return impl_->constraint_scale;
}

const std::array<CylinderForceLimit, crane_model::kActuatedDof> &
Ocp::cylinder_force_max() const noexcept
{
  return impl_->cylinder_force_max;
}

double Ocp::pump_flow_max() const noexcept
{
  return impl_->flow_max;
}

Result<std::vector<double>> Ocp::stage_residual(
  const crane_model::State & x, const OcpOnlyState & ocp_only, const crane_model::Input & u,
  double progress_accel, const OcpStage & stage) const
{
  const std::vector<double> state = reduce(x, ocp_only);
  const std::vector<double> input = reduce_input(u, progress_accel);
  ParameterVector parameter = parameter_vector(x, impl_->payload);
  write_reference(parameter, stage);
  const double time = 0.0;
  std::vector<double> rows(static_cast<std::size_t>(kNy), 0.0);
  if (!crane_ocp::evaluate(
      impl_->backend.residual, {state.data(), input.data(), nullptr, &time, parameter.data()},
      rows.data()))
  {
    return Result<std::vector<double>>::failure(
      failure(
        ErrorCode::SymbolicBackendFailure, "the stage residual of mpc 2 could not be evaluated"));
  }
  return Result<std::vector<double>>::success(std::move(rows));
}

Result<std::vector<double>> Ocp::terminal_residual(
  const crane_model::State & x, const OcpOnlyState & ocp_only, const OcpStage & stage) const
{
  const std::vector<double> state = reduce(x, ocp_only);
  ParameterVector parameter = parameter_vector(x, impl_->payload);
  write_reference(parameter, stage);
  const double time = 0.0;
  std::vector<double> rows(static_cast<std::size_t>(kNyTerminal), 0.0);
  if (!crane_ocp::evaluate(
      impl_->backend.terminal, {state.data(), nullptr, nullptr, &time, parameter.data()},
      rows.data()))
  {
    return Result<std::vector<double>>::failure(
      failure(
        ErrorCode::SymbolicBackendFailure,
        "the terminal residual of mpc 2 could not be evaluated"));
  }
  return Result<std::vector<double>>::success(std::move(rows));
}

std::vector<double> Ocp::stage_weights() const
{
  return impl_->weight_diagonal;
}

std::vector<double> Ocp::terminal_weights() const
{
  return impl_->terminal_weight_diagonal;
}

Result<crane_model::State> Ocp::propagate(
  const crane_model::State & x, const crane_model::Input & u, double seconds) const
{
  if (!std::isfinite(seconds) || seconds < 0.0) {
    return Result<crane_model::State>::failure(
      failure(
        ErrorCode::InvalidArgument,
        "the propagation horizon is the measured transport delay and must be finite and >= 0; "
        "zero means no delay has been measured and nothing is propagated, which is honest, "
        "whereas a negative one is a plan for the past"));
  }
  if (!x.allFinite() || !u.allFinite()) {
    return Result<crane_model::State>::failure(
      failure(
        ErrorCode::NonFiniteInput, "the measured state and the applied input must be finite"));
  }
  if (seconds == 0.0) {
    crane_model::State held = x;
    held[static_cast<Eigen::Index>(kStateActuatedVelocity + kToolRow)] = 0.0;
    return Result<crane_model::State>::success(held);
  }

  const double step_ceiling = impl_->settings.sample_time_s;
  const int steps = std::max(1, static_cast<int>(std::ceil(seconds / step_ceiling)));
  const double step = seconds / static_cast<double>(steps);

  crane_model::State state = x;
  state[static_cast<Eigen::Index>(kStateActuatedVelocity + kToolRow)] = 0.0;
  for (int index = 0; index < steps; ++index) {
    auto k1 = dynamics(state, u);
    if (!k1.ok()) {return Result<crane_model::State>::failure(k1.status());}
    auto k2 = dynamics(crane_model::State(state + 0.5 * step * k1.value()), u);
    if (!k2.ok()) {return Result<crane_model::State>::failure(k2.status());}
    auto k3 = dynamics(crane_model::State(state + 0.5 * step * k2.value()), u);
    if (!k3.ok()) {return Result<crane_model::State>::failure(k3.status());}
    auto k4 = dynamics(crane_model::State(state + step * k3.value()), u);
    if (!k4.ok()) {return Result<crane_model::State>::failure(k4.status());}
    state += (step / 6.0) *
      (k1.value() + 2.0 * k2.value() + 2.0 * k3.value() + k4.value());
  }
  if (!state.allFinite()) {
    return Result<crane_model::State>::failure(
      failure(
        ErrorCode::SymbolicBackendFailure,
        "propagating the measured state forward under the applied command left it non-finite"));
  }
  return Result<crane_model::State>::success(state);
}

Result<InitialGuess> Ocp::cold_start(const crane_model::State & x0) const
{
  if (!x0.allFinite()) {
    return Result<InitialGuess>::failure(
      failure(ErrorCode::NonFiniteInput, "x_0 is not finite, so there is nothing to roll out"));
  }

  const std::size_t intervals = static_cast<std::size_t>(impl_->intervals());
  InitialGuess guess;
  guess.states.assign(intervals + 1U, x0);
  guess.inputs.assign(intervals, crane_model::Input::Zero());
  for (std::size_t stage = 0; stage < intervals; ++stage) {
    auto next = propagate(
      guess.states[stage], crane_model::Input::Zero(), impl_->settings.sample_time_s);
    if (!next.ok()) {
      return Result<InitialGuess>::failure(next.status());
    }
    guess.states[stage + 1U] = std::move(next).value();
  }
  return Result<InitialGuess>::success(std::move(guess));
}

InitialGuess shifted(const OcpSolution & previous)
{
  InitialGuess guess;
  if (previous.states.size() < 2U || previous.inputs.empty()) {
    return guess;
  }

  const std::size_t nodes = previous.states.size();
  guess.states.resize(nodes);
  for (std::size_t stage = 0; stage + 1U < nodes; ++stage) {
    guess.states[stage] = previous.states[stage + 1U];
  }
  guess.states.back() = previous.states.back();

  const std::size_t intervals = previous.inputs.size();
  guess.inputs.resize(intervals);
  for (std::size_t stage = 0; stage + 1U < intervals; ++stage) {
    guess.inputs[stage] = previous.inputs[stage + 1U];
  }
  guess.inputs.back() = previous.inputs.back();

  // The OCP-only rows shift with the states they belong to; leaving them out
  // would hand the next solve a warm start that is cold on the actuator alone.
  if (previous.ocp_only.size() == nodes) {
    guess.ocp_only.resize(nodes);
    for (std::size_t stage = 0; stage + 1U < nodes; ++stage) {
      guess.ocp_only[stage] = previous.ocp_only[stage + 1U];
    }
    guess.ocp_only.back() = previous.ocp_only.back();
    // `s` restarts at zero every cycle and the caller advances the reference
    // origin by `progress_advance` instead, so the carried plan has to be
    // re-expressed against the new origin or the warm start would be a horizon
    // whose progress rows all sit one interval in the past.
    for (OcpOnlyState & node : guess.ocp_only) {
      node.progress = std::max(0.0, node.progress - previous.progress_advance);
    }
  }
  if (previous.progress_accel.size() == intervals) {
    guess.progress_accel.resize(intervals);
    for (std::size_t stage = 0; stage + 1U < intervals; ++stage) {
      guess.progress_accel[stage] = previous.progress_accel[stage + 1U];
    }
    guess.progress_accel.back() = previous.progress_accel.back();
  }
  return guess;
}

Result<OcpSolution> Ocp::solve(
  const crane_model::State & x0, const std::vector<OcpStage> & stages,
  const InitialGuess & guess)
{
  const int intervals = impl_->intervals();
  if (stages.size() != static_cast<std::size_t>(intervals) + 1U) {
    return Result<OcpSolution>::failure(
      failure(
        ErrorCode::InvalidArgument,
        "the reference must carry one stage per shooting node, N + 1 = " +
        std::to_string(intervals + 1) + ", and it carries " + std::to_string(stages.size())));
  }
  if (!x0.allFinite()) {
    return Result<OcpSolution>::failure(
      failure(ErrorCode::NonFiniteInput, "x_0 is not finite"));
  }
  for (const OcpStage & stage : stages) {
    // `ddq_a_ref` and `progress_nominal` are checked here for the same reason the
    // other three are, and it is sharper for the curvature: it is the one row in
    // the chain a short reference segment can destroy -- the resample's Hermite
    // second derivative carries `1/dt^2` in the *incoming* knot spacing, which
    // nothing bounds from below -- and it reaches every stage of the horizon
    // through `p`, multiplied by `ds^2`, into a Gauss-Newton Hessian.
    if (!all_finite(stage.q_a_ref) || !all_finite(stage.dq_a_ref) ||
      !all_finite(stage.ddq_a_ref) || !all_finite(stage.q_eq) ||
      !std::isfinite(stage.progress_nominal))
    {
      return Result<OcpSolution>::failure(
        failure(
          ErrorCode::NonFiniteInput,
          "the reference, its curvature, its expansion point or an equilibrium is not finite"));
    }
  }

  const BoxLimits & limits = impl_->settings.limits;

  const double q_tool = pinned_tool(x0);
  ParameterVector parameter = parameter_vector(x0, impl_->payload);

  // C3's two actuator states are **not measured**, and this is the one decision
  // about them: they are seeded from the model on the first solve and on a
  // payload change, and carried forward from the previous solution's stage 1
  // after that. Carrying forward is what makes them a state rather than a
  // quantity re-derived every cycle -- the force state is real and persists --
  // and re-seeding on a payload change is because `h_eff` moves discontinuously
  // at a grasp, which is the same reason the warm start is dropped there.
  if (!impl_->ocp_only_seeded || impl_->payload_changed) {
    if (impl_->model.has_value()) {
      impl_->ocp_only = seed_ocp_only(*impl_->model, x0, impl_->payload, impl_->ocp_only);
    }
    impl_->ocp_only_seeded = true;
  }

  // **The progress restart** (`timber_crane_mpc.cpp:173-181`). `s` is pinned at
  // zero at stage 0 on every cycle; what the last cycle bought is carried by the
  // caller advancing the reference origin by `progress_advance` instead. Letting
  // `s` accumulate would make the expansion point of every stage's reference
  // drift away from zero over a long move for no gain in what is expressed.
  // The **rate** is carried forward, because it is the real state here: it is
  // what says how fast the plan is currently being spent.
  impl_->ocp_only.progress = 0.0;

  // **The effort term of `wiki/mpc.md` 2 references `h_eff`, not zero** (issue
  // 117). 2 prices `tau_a` rather than `u` so that the price of an acceleration
  // carries the effective inertia, and `tau_a = M_eff ddq_a + h_eff`: against a
  // zero reference the residual is the whole force, so the cost prices the
  // machine holding its own weight and the optimizer buys droop to avoid paying
  // -- measured at 0.53 rad of final error on the offline A2B, with the sway box
  // saturating as a consequence. Against `h_eff` the residual is `M_eff ddq_a`,
  // which is the quantity 2 says it is pricing, and the same move finishes at
  // 0.14 rad.
  //
  // One evaluation per cycle, at `x_0`, held across the horizon -- the trade
  // `mpc_node.cpp` already makes for `q_eq`. Per-stage exactness was measured
  // and buys nothing: 0.15 rad against 0.14.
  if (impl_->model.has_value()) {
    impl_->force_reference =
      static_hold_force(*impl_->model, x0, impl_->payload, impl_->force_reference);
  }
  const std::vector<double> reduced_x0 = reduce(x0, impl_->ocp_only);

  for (int stage = 0; stage <= intervals; ++stage) {
    const OcpStage & node = stages[static_cast<std::size_t>(stage)];

    // The reference travels in `p`, not in `yref`. `q_a,ref(s)` is a function of
    // the progress state and acados subtracts `yref` as a constant, so the
    // tracking rows carry their own reference inside the residual and their
    // `yref` is zero. That also leaves exactly one home for the reference.
    write_reference(parameter, node);
    static_cast<void>(impl_->acados->set_parameters(stage, parameter.data(), kNp));

    std::vector<double> & reference =
      stage < intervals ? impl_->reference : impl_->terminal_reference;
    std::fill(reference.begin(), reference.end(), 0.0);
    // The progress row is a regulator toward one second of plan per second of
    // wall clock -- a quadratic on the rate, not a linear reward on progress.
    // That is what makes the nominal behaviour exactly time-indexed tracking and
    // deviation something the optimizer has to buy.
    reference[kResidualProgressRate] = kProgressRateReference;
    if (stage < intervals) {
      for (std::size_t row = 0; row < kPlannedDof; ++row) {
        reference[kResidualActuatedForce + row] = impl_->force_reference[row];
      }
    }
    for (std::size_t row = 0; row < crane_model::kPassiveDof; ++row) {
      reference[kResidualPassivePosition + row] = node.q_eq[row];
    }
    impl_->acados->set_cost(stage, "yref", reference.data());

    if (stage == 0) {
      for (std::size_t row = 0; row < kOcpStateDof; ++row) {
        impl_->lower_state[row] = reduced_x0[row];
        impl_->upper_state[row] = reduced_x0[row];
      }
    } else {
      for (std::size_t row = 0; row < kPlannedDof; ++row) {
        // Constraint 1, `q_a_margin` already subtracted. The margin is the
        // iLQR's `joint_limit.safety_thresh` (issue 117): there it started a
        // penalty biting early, here it tightens a bound that is kept exactly.
        const auto box = position_box(
          limits.q_a_lower[row], limits.q_a_upper[row], limits.q_a_margin[row],
          reduced_x0[kReducedActuatedPosition + row]);
        impl_->lower_state[kReducedActuatedPosition + row] = box.first;
        impl_->upper_state[kReducedActuatedPosition + row] = box.second;
        impl_->lower_state[kReducedActuatedVelocity + row] = -limits.dq_a_max[row];
        impl_->upper_state[kReducedActuatedVelocity + row] = limits.dq_a_max[row];
      }
      for (std::size_t row = 0; row < crane_model::kPassiveDof; ++row) {
        impl_->lower_state[kReducedPassivePosition + row] = node.q_eq[row] - limits.q_u_max[row];
        impl_->upper_state[kReducedPassivePosition + row] = node.q_eq[row] + limits.q_u_max[row];
        impl_->lower_state[kReducedPassiveVelocity + row] = -limits.dq_u_max[row];
        impl_->upper_state[kReducedPassiveVelocity + row] = limits.dq_u_max[row];
      }
      // `u_f` is a filtered `u`, so it gets `u`'s own box. The force states are
      // not boxed at all and stop the array one block short of `x` -- acados
      // reads `nbx` entries off these pointers, which is why one buffer serves
      // both the pinned stage 0 and the boxed stages after it.
      for (std::size_t slot = 0; slot < kOcpCommandLagDof; ++slot) {
        const std::size_t axis = kCommandLagAxes[slot];
        impl_->lower_state[kReducedCommandLag + slot] = -limits.u_max[axis];
        impl_->upper_state[kReducedCommandLag + slot] = limits.u_max[axis];
      }
      // The progress pair. The rate's floor is zero because the machine does not
      // run the plan backwards; the parameter's own box follows from that floor
      // and the ceiling and is not a second knob. It is **per stage** and it is
      // the exactly reachable set: `s` starts each cycle at zero, so by stage `k`
      // it cannot have outrun `k T_s` spent at the fastest rate allowed. A single
      // horizon-wide bound would leave the early stages free to place `s` far
      // from their own expansion point, where the second-order model of the
      // reference is a curve that was never planned.
      impl_->lower_state[kReducedProgress] = 0.0;
      impl_->upper_state[kReducedProgress] =
        static_cast<double>(stage) * impl_->settings.sample_time_s * limits.progress_rate_max;
      impl_->lower_state[kReducedProgressRate] = 0.0;
      impl_->upper_state[kReducedProgressRate] = limits.progress_rate_max;
    }
    impl_->acados->set_constraint(stage, "lbx", impl_->lower_state.data());
    impl_->acados->set_constraint(stage, "ubx", impl_->upper_state.data());
  }

  impl_->acados->reset();

  const bool payload_changed = impl_->payload_changed;
  impl_->payload_changed = false;

  const std::size_t nodes = static_cast<std::size_t>(intervals) + 1U;
  const bool warm = !payload_changed && guess.warm() && guess.states.size() == nodes &&
    std::all_of(
    guess.states.begin(), guess.states.end(),
    [](const crane_model::State & state) {return state.allFinite();}) &&
    std::all_of(
    guess.inputs.begin(), guess.inputs.end(),
    [](const crane_model::Input & input) {return input.allFinite();});

  std::vector<crane_model::State> initial_state;
  std::vector<crane_model::Input> initial_input;
  if (warm) {
    initial_state = guess.states;
    initial_input = guess.inputs;
  } else {
    auto rollout = cold_start(x0);
    if (rollout.ok()) {
      initial_state = std::move(rollout).value().states;
      initial_input.assign(static_cast<std::size_t>(intervals), crane_model::Input::Zero());
    } else {
      initial_state.assign(nodes, x0);
      initial_input.assign(static_cast<std::size_t>(intervals), crane_model::Input::Zero());
    }
  }

  std::vector<double> rest(static_cast<std::size_t>(kNu), 0.0);
  for (int stage = 0; stage <= intervals; ++stage) {
    const OcpStage & node = stages[static_cast<std::size_t>(stage)];
    const OcpOnlyState & guessed =
      guess.ocp_only.size() == nodes ? guess.ocp_only[static_cast<std::size_t>(stage)]
      : impl_->ocp_only;
    impl_->guess = reduce(initial_state[static_cast<std::size_t>(stage)], guessed);
    if (stage < intervals) {
      const double accel = guess.progress_accel.size() == static_cast<std::size_t>(intervals) ?
        guess.progress_accel[static_cast<std::size_t>(stage)] : 0.0;
      const std::vector<double> applied =
        reduce_input(initial_input[static_cast<std::size_t>(stage)], accel);
      for (std::size_t row = 0; row < kOcpPlannedInputDof; ++row) {
        rest[row] = std::clamp(applied[row], -limits.u_max[row], limits.u_max[row]);
      }
      rest[kInputProgressAccel] = std::clamp(
        std::isfinite(applied[kInputProgressAccel]) ? applied[kInputProgressAccel] : 0.0,
        -limits.progress_accel_max, limits.progress_accel_max);
    }
    if (stage > 0) {
      for (std::size_t row = 0; row < kPlannedDof; ++row) {
        const auto box = position_box(
          limits.q_a_lower[row], limits.q_a_upper[row], limits.q_a_margin[row],
          reduced_x0[kReducedActuatedPosition + row]);
        impl_->guess[kReducedActuatedPosition + row] =
          std::clamp(impl_->guess[kReducedActuatedPosition + row], box.first, box.second);
        impl_->guess[kReducedActuatedVelocity + row] = std::clamp(
          impl_->guess[kReducedActuatedVelocity + row], -limits.dq_a_max[row],
          limits.dq_a_max[row]);
      }
      for (std::size_t row = 0; row < crane_model::kPassiveDof; ++row) {
        impl_->guess[kReducedPassivePosition + row] = std::clamp(
          impl_->guess[kReducedPassivePosition + row], node.q_eq[row] - limits.q_u_max[row],
          node.q_eq[row] + limits.q_u_max[row]);
        impl_->guess[kReducedPassiveVelocity + row] = std::clamp(
          impl_->guess[kReducedPassiveVelocity + row], -limits.dq_u_max[row],
          limits.dq_u_max[row]);
      }
      for (std::size_t slot = 0; slot < kOcpCommandLagDof; ++slot) {
        const std::size_t axis = kCommandLagAxes[slot];
        impl_->guess[kReducedCommandLag + slot] = std::clamp(
          impl_->guess[kReducedCommandLag + slot], -limits.u_max[axis], limits.u_max[axis]);
      }
      impl_->guess[kReducedProgress] = std::clamp(
        impl_->guess[kReducedProgress], 0.0,
        static_cast<double>(stage) * impl_->settings.sample_time_s *
        limits.progress_rate_max);
      impl_->guess[kReducedProgressRate] =
        std::clamp(impl_->guess[kReducedProgressRate], 0.0, limits.progress_rate_max);
    }
    impl_->acados->set_iterate(stage, "x", impl_->guess.data());
    if (stage < intervals) {
      impl_->acados->set_iterate(stage, "u", rest.data());
    }
    if (!impl_->slack_rest.empty()) {
      impl_->acados->set_iterate(stage, "sl", impl_->slack_rest.data());
      impl_->acados->set_iterate(stage, "su", impl_->slack_rest.data());
    }
  }

  const int status = impl_->acados->solve();

  OcpSolution solution;
  solution.status = status;
  solution.status_word = crane_ocp::status_word(status);
  solution.warm_started = warm;
  impl_->acados->get_statistic("sqp_iter", &solution.iterations);
  impl_->acados->get_statistic("time_tot", &solution.timing.total);
  impl_->acados->get_statistic("time_sim_ad", &solution.timing.integration);
  impl_->acados->get_statistic("time_lin", &solution.timing.linearisation);
  impl_->acados->get_statistic("time_qp_sol", &solution.timing.qp);
  solution.solve_time_s = solution.timing.total;
  impl_->acados->get_statistic("qp_status", &solution.qp_status);
  impl_->acados->get_statistic("qp_iter", &solution.qp_iterations);
  solution.budget_exceeded = solution.solve_time_s > impl_->settings.solve_budget_s;

  solution.states.resize(static_cast<std::size_t>(intervals) + 1U);
  solution.ocp_only.resize(static_cast<std::size_t>(intervals) + 1U);
  solution.inputs.resize(static_cast<std::size_t>(intervals));
  solution.progress_accel.resize(static_cast<std::size_t>(intervals));
  std::vector<double> row(static_cast<std::size_t>(kNx), 0.0);
  std::vector<double> input(static_cast<std::size_t>(kNu), 0.0);
  for (int stage = 0; stage < intervals; ++stage) {
    impl_->acados->get_iterate(stage, "u", input.data());
    solution.inputs[static_cast<std::size_t>(stage)] = expand_input(input);
    solution.progress_accel[static_cast<std::size_t>(stage)] = input[kInputProgressAccel];
  }
  for (int stage = 0; stage <= intervals; ++stage) {
    impl_->acados->get_iterate(stage, "x", row.data());
    solution.states[static_cast<std::size_t>(stage)] = expand(row, q_tool);
    // The terminal node has no input, so the axes without a lag state take the
    // last one applied; nothing reads that entry there.
    const std::size_t applied =
      static_cast<std::size_t>(stage < intervals ? stage : intervals - 1);
    impl_->acados->get_iterate(static_cast<int>(applied), "u", input.data());
    solution.ocp_only[static_cast<std::size_t>(stage)] = expand_ocp_only(row, input);
  }
  solution.u0 = solution.inputs.front();
  // How far this horizon spends the plan over the interval that is about to be
  // applied. `T_s` at nominal rate, less when the optimizer has bought time.
  //
  // **Clamped, and not because the number is untrusted in principle.** Under RTI
  // this is the QP's own iterate at node one, so it satisfies the *linearised*
  // dynamics rather than the integrated ones; a badly conditioned step can put it
  // outside what `0 <= v_s <= v_s^max` makes reachable over one interval. The
  // consumer advances the reference origin by it, and an origin that jumped would
  // skip plan the machine never tracked -- a silent one, since every state stays
  // finite. The clamp is the same two numbers the box carries and adds no knob.
  const double advance =
    solution.ocp_only.size() > 1U ? solution.ocp_only[1].progress : 0.0;
  const double advance_ceiling = impl_->settings.sample_time_s * limits.progress_rate_max;
  solution.progress_advance =
    std::isfinite(advance) ? std::clamp(advance, 0.0, advance_ceiling) : 0.0;

  solution.slack.assign(static_cast<std::size_t>(intervals) + 1U, ConstraintSlack{});
  std::vector<double> lower_slack(impl_->slack_rest.size(), 0.0);
  std::vector<double> upper_slack(impl_->slack_rest.size(), 0.0);
  for (int stage = 0; stage <= intervals; ++stage) {
    const std::size_t count =
      static_cast<std::size_t>(impl_->slack_count[static_cast<std::size_t>(stage)]);
    if (count == 0U) {
      continue;
    }
    impl_->acados->get_iterate(stage, "sl", lower_slack.data());
    impl_->acados->get_iterate(stage, "su", upper_slack.data());
    const std::vector<double> & price = stage == 0 ?
      impl_->slack_price_initial :
      (stage < intervals ? impl_->slack_price_path : impl_->slack_price_terminal);
    for (std::size_t entry = 0; entry < count; ++entry) {
      solution.slack_penalty +=
        price[entry] * (std::max(0.0, lower_slack[entry]) + std::max(0.0, upper_slack[entry]));
    }

    ConstraintSlack & taken = solution.slack[static_cast<std::size_t>(stage)];
    const auto magnitude = [&](std::size_t entry) {
        return std::max(std::max(0.0, lower_slack[entry]), std::max(0.0, upper_slack[entry]));
      };
    std::size_t entry = 0U;
    if (stage > 0) {
      for (std::size_t index = 0; index < crane_model::kPassiveDof; ++index) {
        taken.q_u[index] = magnitude(entry++);
      }
      for (std::size_t index = 0; index < crane_model::kPassiveDof; ++index) {
        taken.dq_u[index] = magnitude(entry++);
      }
    }
    if (stage < intervals) {
      for (std::size_t index = 0; index < kPlannedDof; ++index) {
        taken.cylinder_force[index] = magnitude(entry++);
      }
      taken.pump_flow = magnitude(entry++);
    }
    if (taken.worst() > kSlackNoticeable) {
      solution.used_slack = true;
    }

    for (std::size_t index = 0; index < crane_model::kPassiveDof; ++index) {
      solution.violation.q_u =
        std::max(solution.violation.q_u, taken.q_u[index] / limits.q_u_max[index]);
      solution.violation.dq_u =
        std::max(solution.violation.dq_u, taken.dq_u[index] / limits.dq_u_max[index]);
    }
    for (std::size_t index = 0; index < kPlannedDof; ++index) {
      solution.violation.cylinder_force =
        std::max(solution.violation.cylinder_force, taken.cylinder_force[index]);
    }
    solution.violation.pump_flow = std::max(solution.violation.pump_flow, taken.pump_flow);
  }

  solution.dq_a_command =
    x0.segment(static_cast<Eigen::Index>(kStateActuatedVelocity), crane_model::kActuatedDof) +
    impl_->settings.sample_time_s * solution.u0;
  solution.dq_a_command[static_cast<Eigen::Index>(kToolRow)] = 0.0;

  bool finite = solution.u0.allFinite();
  for (const crane_model::State & state : solution.states) {
    finite = finite && state.allFinite();
  }
  if (!finite) {
    solution.outcome = SolveOutcome::Failed;
    if (solution.status == ACADOS_SUCCESS) {
      solution.status_word += " (non-finite horizon)";
    }
  } else if (solution.status != ACADOS_SUCCESS) {
    solution.outcome = SolveOutcome::Failed;
  } else if (solution.qp_status != ACADOS_SUCCESS) {
    solution.outcome = SolveOutcome::Failed;
    solution.status_word += " (QP status " + std::to_string(solution.qp_status) + ")";
  } else if (solution.budget_exceeded) {
    solution.outcome = SolveOutcome::BudgetExceeded;
  } else {
    solution.outcome = SolveOutcome::Converged;
  }

  // Carry the actuator state one interval forward, which is where the next cycle
  // starts. A failed solve leaves the previous estimate in place rather than
  // adopting a horizon nobody accepted.
  if (solution.outcome != SolveOutcome::Failed && solution.ocp_only.size() > 1U) {
    const OcpOnlyState & next = solution.ocp_only[1];
    const auto usable = [](const std::array<double, kPlannedDof> & values) {
        return std::all_of(
          values.begin(), values.end(), [](double value) {return std::isfinite(value);});
      };
    if (usable(next.command_lag) && usable(next.force) && std::isfinite(next.progress) &&
      std::isfinite(next.progress_rate))
    {
      impl_->ocp_only = next;
    }
  }

  return Result<OcpSolution>::success(std::move(solution));
}

Result<CostTerms> Ocp::cost_terms(
  const OcpSolution & solution, const std::vector<OcpStage> & stages) const
{
  const std::size_t intervals = static_cast<std::size_t>(impl_->intervals());
  if (solution.states.size() != intervals + 1U || solution.inputs.size() != intervals ||
    stages.size() != intervals + 1U)
  {
    return Result<CostTerms>::failure(
      failure(
        ErrorCode::InvalidArgument,
        "the cost of mpc 2 is only defined for a solution of this problem against the stages it "
        "was solved on: N + 1 = " + std::to_string(intervals + 1) + " states and stages and N "
        "inputs"));
  }

  const std::vector<double> weights = stage_weights();
  const std::vector<double> terminal = terminal_weights();
  CostTerms terms;
  const auto add = [](double weight, double residual, double reference) {
      const double error = residual - reference;
      return 0.5 * weight * error * error;
    };

  const bool carried = solution.ocp_only.size() == intervals + 1U &&
    solution.progress_accel.size() == intervals;
  if (!carried) {
    return Result<CostTerms>::failure(
      failure(
        ErrorCode::InvalidArgument,
        "the cost of mpc 2 needs the solution's OCP-only rows and its progress input: the "
        "reference is evaluated at the progress state, so a residual re-evaluated without them "
        "would price a different problem than the solver minimised"));
  }

  for (std::size_t stage = 0; stage < intervals; ++stage) {
    auto y = stage_residual(
      solution.states[stage], solution.ocp_only[stage], solution.inputs[stage],
      solution.progress_accel[stage], stages[stage]);
    if (!y.ok()) {
      return Result<CostTerms>::failure(y.status());
    }
    const std::vector<double> & row = y.value();
    if (row.size() != static_cast<std::size_t>(kNy)) {
      return Result<CostTerms>::failure(
        failure(
          ErrorCode::SymbolicBackendFailure,
          "the stage residual of mpc 2 came back the wrong length"));
    }
    const OcpStage & node = stages[stage];
    // The tracking rows carry their own reference, so their `yref` is zero here
    // exactly as it is in the solve. Pricing them against `q_a_ref` a second
    // time would double-count the reference and report a term the solver never
    // minimised.
    for (std::size_t index = 0; index < kPlannedDof; ++index) {
      terms.q_a += add(
        weights[kResidualActuatedPosition + index], row[kResidualActuatedPosition + index], 0.0);
      terms.dq_a += add(
        weights[kResidualActuatedVelocity + index], row[kResidualActuatedVelocity + index], 0.0);
      terms.tau_a += add(
        weights[kResidualActuatedForce + index], row[kResidualActuatedForce + index],
        impl_->force_reference[index]);
      terms.u += add(weights[kResidualInput + index], row[kResidualInput + index], 0.0);
    }
    terms.u += add(
      weights[kResidualInput + kInputProgressAccel], row[kResidualInput + kInputProgressAccel],
      0.0);
    terms.lag += add(weights[kResidualLag], row[kResidualLag], 0.0);
    terms.progress += add(
      weights[kResidualProgressRate], row[kResidualProgressRate], kProgressRateReference);
    for (std::size_t index = 0; index < crane_model::kPassiveDof; ++index) {
      terms.q_u += add(
        weights[kResidualPassivePosition + index], row[kResidualPassivePosition + index],
        node.q_eq[index]);
      terms.dq_u += add(
        weights[kResidualPassiveVelocity + index], row[kResidualPassiveVelocity + index], 0.0);
    }
  }

  auto terminal_row = terminal_residual(
    solution.states[intervals], solution.ocp_only[intervals], stages[intervals]);
  if (!terminal_row.ok()) {
    return Result<CostTerms>::failure(terminal_row.status());
  }
  if (terminal_row.value().size() != static_cast<std::size_t>(kNyTerminal)) {
    return Result<CostTerms>::failure(
      failure(
        ErrorCode::SymbolicBackendFailure,
        "the terminal residual of mpc 2 came back the wrong length"));
  }
  const OcpStage & last = stages[intervals];
  for (std::size_t index = 0; index < kPlannedDof; ++index) {
    terms.terminal += add(
      terminal[kResidualActuatedPosition + index],
      terminal_row.value()[kResidualActuatedPosition + index], 0.0);
    terms.terminal += add(
      terminal[kResidualActuatedVelocity + index],
      terminal_row.value()[kResidualActuatedVelocity + index], 0.0);
  }
  terms.terminal += add(
    terminal[kResidualLag], terminal_row.value()[kResidualLag], 0.0);
  terms.terminal += add(
    terminal[kResidualProgressRate], terminal_row.value()[kResidualProgressRate],
    kProgressRateReference);
  for (std::size_t index = 0; index < crane_model::kPassiveDof; ++index) {
    terms.terminal += add(
      terminal[kResidualPassivePosition + index],
      terminal_row.value()[kResidualPassivePosition + index], last.q_eq[index]);
    terms.terminal += add(
      terminal[kResidualPassiveVelocity + index],
      terminal_row.value()[kResidualPassiveVelocity + index], 0.0);
  }

  terms.slack = solution.slack_penalty;
  return Result<CostTerms>::success(terms);
}

}  // namespace crane_mpc
