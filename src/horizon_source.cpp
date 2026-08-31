// Reference resampling and horizon message conversion.
#include "crane_mpc/horizon_source.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "builtin_interfaces/msg/duration.hpp"

namespace crane_mpc
{
namespace
{

constexpr std::size_t kNoColumn = std::numeric_limits<std::size_t>::max();

constexpr double kNanosecondsPerSecond = 1.0e9;

double duration_seconds(const builtin_interfaces::msg::Duration & duration)
{
  return static_cast<double>(duration.sec) +
         static_cast<double>(duration.nanosec) / kNanosecondsPerSecond;
}

builtin_interfaces::msg::Duration seconds_duration(double seconds)
{
  builtin_interfaces::msg::Duration duration;
  const double whole = std::floor(seconds);
  double nanoseconds = std::round((seconds - whole) * kNanosecondsPerSecond);
  double sec = whole;
  if (nanoseconds >= kNanosecondsPerSecond) {
    nanoseconds -= kNanosecondsPerSecond;
    sec += 1.0;
  }
  duration.sec = static_cast<std::int32_t>(sec);
  duration.nanosec = static_cast<std::uint32_t>(nanoseconds);
  return duration;
}

bool finite(const std::array<double, crane_model::kActuatedDof> & row)
{
  return std::all_of(row.begin(), row.end(), [](double value) {return std::isfinite(value);});
}

void hermite(
  const Knot & left, const Knot & right, double t,
  std::array<double, crane_model::kActuatedDof> & q_a_ref,
  std::array<double, crane_model::kActuatedDof> & dq_a_ref)
{
  const double dt = right.t - left.t;
  const double s = (t - left.t) / dt;
  const double s2 = s * s;
  const double s3 = s2 * s;

  const double h00 = 2.0 * s3 - 3.0 * s2 + 1.0;
  const double h10 = s3 - 2.0 * s2 + s;
  const double h01 = -2.0 * s3 + 3.0 * s2;
  const double h11 = s3 - s2;

  const double d00 = 6.0 * s2 - 6.0 * s;
  const double d10 = 3.0 * s2 - 4.0 * s + 1.0;
  const double d01 = -6.0 * s2 + 6.0 * s;
  const double d11 = 3.0 * s2 - 2.0 * s;

  for (std::size_t axis = 0; axis < crane_model::kActuatedDof; ++axis) {
    const double q0 = left.q_a_ref[axis];
    const double q1 = right.q_a_ref[axis];
    const double v0 = left.dq_a_ref[axis];
    const double v1 = right.dq_a_ref[axis];
    q_a_ref[axis] = h00 * q0 + h10 * dt * v0 + h01 * q1 + h11 * dt * v1;
    dq_a_ref[axis] = (d00 * q0 + d01 * q1) / dt + d10 * v0 + d11 * v1;
  }
}

}

const char * to_string(ResampleRejection rejection)
{
  switch (rejection) {
    case ResampleRejection::None:
      return "accepted";
    case ResampleRejection::EmptyReference:
      return "the reference carries no points";
    case ResampleRejection::DegenerateGrid:
      return "the horizon grid is shorter than two knots or its step is not positive";
    case ResampleRejection::NonMonotonicTime:
      return "the reference's times are not strictly increasing";
    case ResampleRejection::NonFinite:
      return "the reference carries a value that is not finite";
    case ResampleRejection::OffsetBeforeReference:
      return "the horizon would start before the reference's own first point";
  }
  return "unknown";
}

ResampleRejection resample(
  const std::vector<Knot> & reference, double offset, const HorizonGrid & grid,
  std::vector<Knot> & horizon)
{
  horizon.clear();

  if (grid.horizon_length < 2 || !std::isfinite(grid.Ts) || grid.Ts <= 0.0) {
    return ResampleRejection::DegenerateGrid;
  }
  if (reference.empty()) {
    return ResampleRejection::EmptyReference;
  }
  if (!std::isfinite(offset)) {
    return ResampleRejection::NonFinite;
  }
  for (std::size_t index = 0; index < reference.size(); ++index) {
    const Knot & knot = reference[index];
    if (!std::isfinite(knot.t) || !finite(knot.q_a_ref) || !finite(knot.dq_a_ref)) {
      return ResampleRejection::NonFinite;
    }
    if (index > 0 && !(knot.t > reference[index - 1].t)) {
      return ResampleRejection::NonMonotonicTime;
    }
  }
  if (offset < reference.front().t) {
    return ResampleRejection::OffsetBeforeReference;
  }

  horizon.resize(grid.horizon_length);
  std::size_t segment = 0;
  for (std::size_t index = 0; index < grid.horizon_length; ++index) {
    const double t = offset + static_cast<double>(index) * grid.Ts;
    horizon[index].t = static_cast<double>(index) * grid.Ts;

    if (t >= reference.back().t) {
      horizon[index].q_a_ref = reference.back().q_a_ref;
      horizon[index].dq_a_ref.fill(0.0);
      continue;
    }
    while (segment + 2 < reference.size() && reference[segment + 1].t <= t) {
      ++segment;
    }
    hermite(
      reference[segment], reference[segment + 1], t, horizon[index].q_a_ref,
      horizon[index].dq_a_ref);
  }
  return ResampleRejection::None;
}

bool reference_from_message(
  const trajectory_msgs::msg::JointTrajectory & message,
  const std::array<std::string, crane_model::kActuatedDof> & joints,
  std::vector<Knot> & reference, std::string & why)
{
  reference.clear();
  why.clear();

  if (message.points.empty()) {
    why = "the reference carries no points";
    return false;
  }

  std::array<std::size_t, crane_model::kActuatedDof> column{};
  column.fill(kNoColumn);
  for (std::size_t axis = 0; axis < crane_model::kActuatedDof; ++axis) {
    const auto found =
      std::find(message.joint_names.begin(), message.joint_names.end(), joints[axis]);
    if (found == message.joint_names.end()) {
      why = "the reference does not name " + joints[axis];
      return false;
    }
    column[axis] = static_cast<std::size_t>(std::distance(message.joint_names.begin(), found));
  }

  const std::size_t width = message.joint_names.size();
  std::vector<Knot> read(message.points.size());
  for (std::size_t index = 0; index < message.points.size(); ++index) {
    const auto & point = message.points[index];
    const std::string at = " at point " + std::to_string(index);
    if (point.positions.size() != width) {
      why = "the reference carries no position for every joint" + at;
      return false;
    }
    if (point.velocities.size() != width) {
      why = "the reference carries no velocity for every joint" + at;
      return false;
    }
    read[index].t = duration_seconds(point.time_from_start);
    for (std::size_t axis = 0; axis < crane_model::kActuatedDof; ++axis) {
      read[index].q_a_ref[axis] = point.positions[column[axis]];
      read[index].dq_a_ref[axis] = point.velocities[column[axis]];
    }
  }
  reference = std::move(read);
  return true;
}

void horizon_to_message(
  const std::vector<Knot> & horizon,
  const std::array<std::string, crane_model::kActuatedDof> & joints,
  const builtin_interfaces::msg::Time & first_knot_valid_at,
  trajectory_msgs::msg::JointTrajectory & message)
{
  message = trajectory_msgs::msg::JointTrajectory{};
  message.header.stamp = first_knot_valid_at;
  message.header.frame_id = "";
  message.joint_names.assign(joints.begin(), joints.end());
  message.points.resize(horizon.size());
  for (std::size_t index = 0; index < horizon.size(); ++index) {
    auto & point = message.points[index];
    point.positions.assign(horizon[index].q_a_ref.begin(), horizon[index].q_a_ref.end());
    point.velocities.assign(horizon[index].dq_a_ref.begin(), horizon[index].dq_a_ref.end());
    point.time_from_start = seconds_duration(horizon[index].t);
  }
}

}
