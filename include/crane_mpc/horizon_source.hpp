
// Horizon construction and ROS trajectory-message conversion.
#ifndef CRANE_MPC__HORIZON_SOURCE_HPP_
#define CRANE_MPC__HORIZON_SOURCE_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "builtin_interfaces/msg/time.hpp"
#include "crane_model/model.hpp"
#include "trajectory_msgs/msg/joint_trajectory.hpp"

namespace crane_mpc
{

inline constexpr char kHorizonTopic[] = "/crane/mpc/horizon";

// How many knots a horizon may carry. crane_mpc's own bound: a receiver that cannot hold the
// horizon must refuse it rather than truncate, so this is checked at construction.
inline constexpr std::size_t kHorizonKnotCapacity = 64;

struct Knot
{
  double t{};
  std::array<double, crane_model::kActuatedDof> q_a_ref{};
  std::array<double, crane_model::kActuatedDof> dq_a_ref{};
  /// The reference's second derivative in its own time parameter.
  /**
   * The MPC evaluates the reference at its progress state rather than at knot
   * time, so each stage needs a second-order expansion of its own reference and
   * not just a value and a slope. The resample already interpolates with a cubic
   * Hermite, so this is the second derivative of the polynomial it is already
   * evaluating -- no new information and no new assumption about the reference.
   *
   * It is not read off the incoming message: `trajectory_msgs` carries an
   * `accelerations` field and the planner does not populate it here, so a
   * reference read from the wire would be second-order only by accident.
   */
  std::array<double, crane_model::kActuatedDof> ddq_a_ref{};
};

struct HorizonGrid
{
  double Ts{0.04};
  std::size_t horizon_length{50};

  [[nodiscard]] double duration() const noexcept
  {
    return horizon_length < 2 ? 0.0 : static_cast<double>(horizon_length - 1) * Ts;
  }
};

enum class ResampleRejection : std::uint8_t
{
  None = 0,
  EmptyReference,
  DegenerateGrid,
  NonMonotonicTime,
  NonFinite,
  OffsetBeforeReference,
};

[[nodiscard]] const char * to_string(ResampleRejection rejection);

[[nodiscard]] ResampleRejection resample(
  const std::vector<Knot> & reference, double offset, const HorizonGrid & grid,
  std::vector<Knot> & horizon);

[[nodiscard]] bool reference_from_message(
  const trajectory_msgs::msg::JointTrajectory & message,
  const std::array<std::string, crane_model::kActuatedDof> & joints,
  std::vector<Knot> & reference, std::string & why);

void horizon_to_message(
  const std::vector<Knot> & horizon,
  const std::array<std::string, crane_model::kActuatedDof> & joints,
  const builtin_interfaces::msg::Time & first_knot_valid_at,
  trajectory_msgs::msg::JointTrajectory & message);

}

#endif
