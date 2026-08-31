
// ROS-independent interface to the generated acados MPC solver.
#ifndef CRANE_MPC__OCP_SOLVER_HPP_
#define CRANE_MPC__OCP_SOLVER_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "crane_model/model.hpp"

namespace crane_mpc
{

inline constexpr std::size_t kStateActuatedPosition = 0;
inline constexpr std::size_t kStatePassivePosition = 6;
inline constexpr std::size_t kStateActuatedVelocity = 8;
inline constexpr std::size_t kStatePassiveVelocity = 14;

inline constexpr std::size_t kToolRow = crane_model::kActuatedDof - 1U;

inline constexpr std::size_t kPlannedDof = crane_model::kActuatedDof - 1U;

inline constexpr std::size_t kOcpStateDof = 2U * kPlannedDof + 2U * crane_model::kPassiveDof;

inline constexpr std::size_t kOcpInputDof = kPlannedDof;

inline constexpr double kSlackNoticeable = 1.0e-6;

struct CostWeights
{
  std::array<double, crane_model::kActuatedDof> q_a{{4.0, 4.0, 4.0, 4.0, 1.0, 1.0}};
  std::array<double, crane_model::kActuatedDof> dq_a{{0.4, 0.4, 0.4, 0.4, 0.1, 0.1}};
  std::array<double, crane_model::kPassiveDof> q_u{{20.0, 20.0}};
  std::array<double, crane_model::kPassiveDof> dq_u{{4.0, 4.0}};
    std::array<double, crane_model::kActuatedDof> tau_a{
    {1.0e-8, 1.0e-8, 1.0e-8, 1.0e-8, 1.0e-8, 1.0e-8}};
  std::array<double, crane_model::kActuatedDof> u{{0.01, 0.01, 0.01, 0.01, 0.01, 0.01}};

    double terminal_scale{10.0};
};

struct BoxLimits
{
  std::array<double, crane_model::kActuatedDof> q_a_lower{
    {-3.71, -1.2, -0.91, 0.0, -12.566, 0.2}};
  std::array<double, crane_model::kActuatedDof> q_a_upper{
    {3.71, 1.563, 4.6, 2.236, 12.566, 0.7}};
  std::array<double, crane_model::kActuatedDof> dq_a_max{
    {0.802, 0.326, 0.344, 0.630, 2.122, 0.5}};
  std::array<double, crane_model::kPassiveDof> q_u_max{{0.2, 0.2}};
  std::array<double, crane_model::kPassiveDof> dq_u_max{{1.0, 0.5}};
  std::array<double, crane_model::kActuatedDof> u_max{{0.5, 0.7, 0.5, 1.0, 6.0, 2.0}};
};

/// `F_i^max` per direction, and it is the model's answer rather than this one.
/**
 * Chamber areas times relief pressure is a force, so the question belongs to
 * whatever holds the description and `config/hydraulics.yaml` -- which is
 * `crane_model`. It was declared here and, identically, in
 * `crane_planning/timing_ocp.hpp`; both now call the model's own
 * `crane_model::derive_cylinder_force_limits`.
 */
using CylinderForceLimit = crane_model::CylinderForceLimit;

struct HydraulicLimits
{
    double pump_flow_max{1.4e-3};

    double pump_flow_planning_factor{0.95};

    double system_pressure_pa{2.5e7};
};

struct SlackWeights
{
  std::array<double, crane_model::kPassiveDof> q_u{{1.0e3, 1.0e3}};

  std::array<double, crane_model::kPassiveDof> dq_u{{1.0e3, 1.0e3}};

    std::array<double, crane_model::kActuatedDof> cylinder_force{
    {1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e4}};

    double pump_flow{1.0e4};
};

struct OcpSettings
{
    double sample_time_s{0.04};

    std::size_t horizon_length{49};

  CostWeights weights{};
  BoxLimits limits{};
  HydraulicLimits hydraulics{};
  SlackWeights slack{};

    double solve_budget_s{0.03};

    double levenberg_marquardt{1.0e-6};
};

struct OcpStage
{
  std::array<double, crane_model::kActuatedDof> q_a_ref{};
  std::array<double, crane_model::kActuatedDof> dq_a_ref{};
  std::array<double, crane_model::kPassiveDof> q_eq{};
};

struct InitialGuess
{
  std::vector<crane_model::State> states;
  std::vector<crane_model::Input> inputs;

    [[nodiscard]] bool warm() const noexcept
  {
    return !states.empty() && states.size() == inputs.size() + 1U;
  }
};

struct ConstraintSlack
{
  std::array<double, crane_model::kPassiveDof> q_u{};
  std::array<double, crane_model::kPassiveDof> dq_u{};
    std::array<double, kPlannedDof> cylinder_force{};
  double pump_flow{};

  [[nodiscard]] double worst() const noexcept;
};

struct ConstraintViolation
{
  double q_u{};
  double dq_u{};
  double cylinder_force{};
  double pump_flow{};

  [[nodiscard]] double worst() const noexcept;
};

enum class SolveOutcome : std::uint8_t
{
  Converged = 0,
  BudgetExceeded = 1,
  Failed = 2,
};

[[nodiscard]] const char * to_string(SolveOutcome outcome) noexcept;

struct CostTerms
{
  double q_a{};
  double dq_a{};
  double q_u{};
  double dq_u{};
  double tau_a{};
  double u{};
  double terminal{};
  double slack{};

  [[nodiscard]] double total() const noexcept;
};

struct SolveTiming
{
  double total{};
  double integration{};
  double linearisation{};
  double qp{};
};

struct OcpSolution
{
    std::vector<crane_model::State> states;

    crane_model::Input u0{crane_model::Input::Zero()};

    std::vector<crane_model::Input> inputs;

    crane_model::DQA dq_a_command{crane_model::DQA::Zero()};

  std::vector<ConstraintSlack> slack;

  ConstraintViolation violation{};

    bool used_slack{false};

    double slack_penalty{};

  double solve_time_s{};

  SolveTiming timing{};

    bool budget_exceeded{false};

    SolveOutcome outcome{SolveOutcome::Failed};

  int status{};
  std::string status_word;

    int qp_status{};
  int qp_iterations{};

  int iterations{};

    bool warm_started{false};
};

[[nodiscard]] InitialGuess shifted(const OcpSolution & previous);

class Ocp
{
public:
    [[nodiscard]] static crane_model::Result<std::unique_ptr<Ocp>> create(
    const crane_model::ModelConfig & config, const crane_model::Payload & payload,
    const OcpSettings & settings);

  ~Ocp();
  Ocp(const Ocp &) = delete;
  Ocp & operator=(const Ocp &) = delete;
  Ocp(Ocp &&) = delete;
  Ocp & operator=(Ocp &&) = delete;

    [[nodiscard]] crane_model::Result<OcpSolution> solve(
    const crane_model::State & x0, const std::vector<OcpStage> & stages,
    const InitialGuess & guess = InitialGuess{});

    [[nodiscard]] crane_model::Result<CostTerms> cost_terms(
    const OcpSolution & solution, const std::vector<OcpStage> & stages) const;

    [[nodiscard]] crane_model::Result<InitialGuess> cold_start(
    const crane_model::State & x0) const;

    [[nodiscard]] crane_model::Result<crane_model::State> propagate(
    const crane_model::State & x, const crane_model::Input & u, double seconds) const;

    [[nodiscard]] crane_model::Result<crane_model::State> dynamics(
    const crane_model::State & x, const crane_model::Input & u) const;

    [[nodiscard]] crane_model::Result<crane_model::Vector6> actuated_force(
    const crane_model::State & x, const crane_model::Input & u) const;

    [[nodiscard]] crane_model::Result<crane_model::Vector6> cylinder_force(
    const crane_model::State & x, const crane_model::Input & u) const;

    [[nodiscard]] crane_model::Result<crane_model::Vector6> axis_flow(
    const crane_model::State & x, const crane_model::Input & u) const;

    [[nodiscard]] crane_model::Result<std::vector<double>> nonlinear_constraint(
    const crane_model::State & x, const crane_model::Input & u) const;

  [[nodiscard]] const std::vector<double> & constraint_lower() const noexcept;
  [[nodiscard]] const std::vector<double> & constraint_upper() const noexcept;

    [[nodiscard]] const std::vector<double> & constraint_scale() const noexcept;

    [[nodiscard]] const std::array<CylinderForceLimit, crane_model::kActuatedDof> &
  cylinder_force_max() const noexcept;

  [[nodiscard]] double pump_flow_max() const noexcept;

    [[nodiscard]] crane_model::Result<std::vector<double>> stage_residual(
    const crane_model::State & x, const crane_model::Input & u) const;
  [[nodiscard]] crane_model::Result<std::vector<double>> terminal_residual(
    const crane_model::State & x) const;

  [[nodiscard]] std::vector<double> stage_weights() const;
  [[nodiscard]] std::vector<double> terminal_weights() const;

  [[nodiscard]] const OcpSettings & settings() const noexcept;

  [[nodiscard]] const crane_model::Payload & payload() const noexcept;

    [[nodiscard]] crane_model::Status set_payload(const crane_model::Payload & payload);

    void set_solve_budget(double seconds) noexcept;

    static constexpr std::size_t kStageResidualDof = 2 * kPlannedDof +
    2 * crane_model::kPassiveDof + 2 * kPlannedDof;
  static constexpr std::size_t kTerminalResidualDof = 2 * kPlannedDof +
    2 * crane_model::kPassiveDof;

    static constexpr std::size_t kNonlinearConstraintDof = kPlannedDof + 1U;
  static constexpr std::size_t kConstraintCylinderForce = 0U;
  static constexpr std::size_t kConstraintPumpFlow = kPlannedDof;

private:
  struct Impl;
  explicit Ocp(std::unique_ptr<Impl> impl) noexcept;

  [[nodiscard]] crane_model::Result<crane_model::Vector6> output_block(
    const crane_model::State & x, const crane_model::Input & u, std::size_t offset) const;

  std::unique_ptr<Impl> impl_;
};

}  // namespace crane_mpc

#endif  // CRANE_MPC__OCP_SOLVER_HPP_
