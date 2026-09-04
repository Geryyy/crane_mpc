// ROS callbacks, MPC execution, and horizon publication.
#include "crane_mpc/mpc_node.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "crane_mpc/crane_mpc_parameters.hpp"
#include "crane_msgs/msg/payload.hpp"
#include "crane_msgs/msg/supervisor_status.hpp"

namespace
{

constexpr int kWarnPeriodMs = 5000;

bool payload_from_message(
  const crane_msgs::msg::Payload & message, crane_model::Payload & payload, std::string & why)
{
  payload = crane_model::Payload{};
  payload.valid = true;
  payload.inertia_k8_kg_m2.setZero();
  if (message.shape == crane_msgs::msg::Payload::SHAPE_NONE) {
    payload.mass_kg = 0.0;
    payload.center_of_mass_k8_m.setZero();
    return true;
  }
  if (!std::isfinite(message.mass) || message.mass <= 0.0) {
    why = "a payload shape was declared with a mass of " + std::to_string(message.mass) +
      " kg; an unknown payload is not a zero-mass payload, so declare SHAPE_NONE for an empty "
      "gripper instead";
    return false;
  }
  if (!std::isfinite(message.com.x) || !std::isfinite(message.com.y) ||
    !std::isfinite(message.com.z))
  {
    why = "the payload's centre of mass carries a number that is not finite; it is the moment arm "
      "the whole gravity load hangs on";
    return false;
  }
  payload.mass_kg = message.mass;
  payload.center_of_mass_k8_m = Eigen::Vector3d(message.com.x, message.com.y, message.com.z);
  return true;
}

constexpr std::array<std::size_t, crane_model::kActuatedDof> kActuatedCanonical{{0, 1, 2, 3, 6, 7}};

// The tip and the tilt, in the index order of Nomenclature 2 -- the same rows `kPassiveRows` names
// in crane_model.
constexpr std::array<std::size_t, crane_model::kPassiveDof> kPassiveCanonical{{4, 5}};

constexpr double kRateTolerance = 1.0e-9;

bool mode_from_string(const std::string & name, crane_mpc::Mode & mode)
{
  if (name == "shadow") {
    mode = crane_mpc::Mode::Shadow;
    return true;
  }
  if (name == "active") {
    mode = crane_mpc::Mode::Active;
    return true;
  }
  return false;
}

std::string as_text(double value)
{
  std::array<char, 32> buffer{};
  std::snprintf(buffer.data(), buffer.size(), "%.17g", value);
  return std::string(buffer.data());
}

void put(
  std::vector<diagnostic_msgs::msg::KeyValue> & values, const std::string & key,
  const std::string & value)
{
  diagnostic_msgs::msg::KeyValue entry;
  entry.key = key;
  entry.value = value;
  values.push_back(std::move(entry));
}

template<std::size_t N>
void copy_rows(const std::vector<double> & from, std::array<double, N> & into)
{
  for (std::size_t row = 0; row < N && row < from.size(); ++row) {
    into[row] = from[row];
  }
}

}

namespace crane_mpc
{

rclcpp::QoS horizon_qos()
{
  return rclcpp::QoS(rclcpp::KeepLast(1)).reliable().durability_volatile();
}

rclcpp::QoS reference_qos()
{
  return rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
}

rclcpp::QoS robot_description_qos()
{
  return rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
}

rclcpp::QoS measurement_qos()
{
  return rclcpp::QoS(rclcpp::KeepLast(1)).reliable().durability_volatile();
}

rclcpp::QoS payload_estimate_qos()
{
  return rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
}

rclcpp::QoS solver_health_qos()
{
  return rclcpp::QoS(rclcpp::KeepLast(1)).reliable().durability_volatile();
}

rclcpp::QoS controller_state_qos()
{
  return rclcpp::QoS(rclcpp::KeepLast(1)).reliable().durability_volatile();
}

rclcpp::QoS shadow_qos()
{
  return rclcpp::QoS(rclcpp::KeepLast(1)).reliable().durability_volatile();
}

const char * to_string(Mode mode)
{
  return mode == Mode::Active ? "active" : "shadow";
}

struct MpcNode::Parameters
{
  explicit Parameters(MpcNode * node)
  : listener(node), values(listener.get_params()) {}

  crane_mpc::ParamListener listener;
  crane_mpc::Params values;
};

MpcNode::~MpcNode() = default;

MpcNode::MpcNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("crane_mpc", options)
{
  parameters_ = std::make_unique<Parameters>(this);
  const crane_mpc::Params & parameters = parameters_->values;

  if (!mode_from_string(parameters.mode, mode_)) {
    throw std::runtime_error("crane_mpc: unknown mode '" + parameters.mode + "'");
  }

  grid_.Ts = parameters.Ts;
  grid_.horizon_length = static_cast<std::size_t>(parameters.horizon_length);
  max_reference_age_ = parameters.max_reference_age;
  max_clock_skew_ = parameters.max_clock_skew;
  max_state_age_ = parameters.max_state_age;
  sensor_to_valve_delay_ = parameters.sensor_to_valve_delay;
  health_decimation_ = static_cast<std::size_t>(parameters.solver_health_decimation);
  max_consecutive_failures_ = static_cast<std::size_t>(parameters.max_consecutive_failures);

  ocp_settings_.sample_time_s = grid_.Ts;
  ocp_settings_.horizon_length = grid_.horizon_length - 1U;
  ocp_settings_.solve_budget_s = parameters.solve_budget;
  ocp_settings_.levenberg_marquardt = parameters.levenberg_marquardt;
  copy_rows(parameters.weights.q_a, ocp_settings_.weights.q_a);
  copy_rows(parameters.weights.dq_a, ocp_settings_.weights.dq_a);
  copy_rows(parameters.weights.q_u, ocp_settings_.weights.q_u);
  copy_rows(parameters.weights.dq_u, ocp_settings_.weights.dq_u);
  copy_rows(parameters.weights.tau_a, ocp_settings_.weights.tau_a);
  copy_rows(parameters.weights.u, ocp_settings_.weights.u);
  ocp_settings_.weights.terminal_scale = parameters.weights.terminal_scale;
  copy_rows(parameters.limits.q_a_lower, ocp_settings_.limits.q_a_lower);
  copy_rows(parameters.limits.q_a_upper, ocp_settings_.limits.q_a_upper);
  copy_rows(parameters.limits.dq_a_max, ocp_settings_.limits.dq_a_max);
  copy_rows(parameters.limits.q_u_max, ocp_settings_.limits.q_u_max);
  copy_rows(parameters.limits.dq_u_max, ocp_settings_.limits.dq_u_max);
  copy_rows(parameters.limits.u_max, ocp_settings_.limits.u_max);
  ocp_settings_.hydraulics.pump_flow_max = parameters.hydraulics.pump_flow_max;
  ocp_settings_.hydraulics.pump_flow_planning_factor =
    parameters.hydraulics.pump_flow_planning_factor;
  ocp_settings_.hydraulics.system_pressure_pa = parameters.hydraulics.system_pressure_pa;
  copy_rows(parameters.slack.q_u, ocp_settings_.slack.q_u);
  copy_rows(parameters.slack.dq_u, ocp_settings_.slack.dq_u);
  copy_rows(parameters.slack.cylinder_force, ocp_settings_.slack.cylinder_force);
  ocp_settings_.slack.pump_flow = parameters.slack.pump_flow;

  payload_.valid = true;
  payload_.mass_kg = 0.0;
  payload_.center_of_mass_k8_m.setZero();
  payload_.inertia_k8_kg_m2.setZero();

  if (grid_.horizon_length > kHorizonKnotCapacity) {
    throw std::runtime_error(
            "crane_mpc: horizon_length is " + std::to_string(grid_.horizon_length) +
            " knots and the receiver holds at most " +
            std::to_string(kHorizonKnotCapacity) +
            " (kHorizonKnotCapacity); a horizon it cannot read is not a horizon");
  }

  if (max_consecutive_failures_ >= grid_.horizon_length) {
    RCLCPP_WARN(
      get_logger(),
      "max_consecutive_failures is %zu against a horizon of %zu knots, so mpc 6's escalation "
      "cannot fire before the fallback has degenerated: after %zu consecutive fallbacks every knot "
      "published is a copy of one knot the optimizer computed that many cycles ago, holding a pose "
      "at a velocity from somewhere else. Below %zu the escalation stops the publisher while what "
      "it stops publishing is still a plan.",
      max_consecutive_failures_, grid_.horizon_length, grid_.horizon_length - 1U,
      grid_.horizon_length);
  }

  const double rate = 1.0 / grid_.Ts;
  if (std::abs(rate - kHorizonRate) > kRateTolerance) {
    RCLCPP_WARN(
      get_logger(),
      "Ts = %g s puts %s at %.4f Hz. ROS 2 Interfaces 4 fixes that row at %.1f Hz and 1 gives the "
      "interface note precedence over the code, so this deployment is running an amended contract "
      "rather than a tuned node.",
      grid_.Ts, kHorizonTopic, rate, kHorizonRate);
  }

  horizon_publisher_ =
    create_publisher<trajectory_msgs::msg::JointTrajectory>(kHorizonTopic, horizon_qos());
  shadow_horizon_publisher_ =
    create_publisher<trajectory_msgs::msg::JointTrajectory>(kShadowHorizonTopic, shadow_qos());
  tcp_horizon_publisher_ = create_publisher<nav_msgs::msg::Path>(kTcpHorizonTopic, horizon_qos());
  health_publisher_ =
    create_publisher<crane_msgs::msg::SolverHealth>(kSolverHealthTopic, solver_health_qos());
  comparison_publisher_ =
    create_publisher<diagnostic_msgs::msg::DiagnosticArray>(kShadowComparisonTopic, shadow_qos());

  controller_state_subscription_ =
    create_subscription<control_msgs::msg::JointTrajectoryControllerState>(
    kControllerStateTopic, controller_state_qos(),
    [this](control_msgs::msg::JointTrajectoryControllerState::ConstSharedPtr message) {
      on_controller_state(std::move(message));
    });
  reference_subscription_ = create_subscription<trajectory_msgs::msg::JointTrajectory>(
    kReferenceTopic, reference_qos(),
    [this](trajectory_msgs::msg::JointTrajectory::ConstSharedPtr message) {
      on_reference(std::move(message));
    });
  robot_description_subscription_ = create_subscription<std_msgs::msg::String>(
    kRobotDescriptionTopic, robot_description_qos(),
    [this](std_msgs::msg::String::ConstSharedPtr message) {on_robot_description(message);});
  joint_state_subscription_ = create_subscription<sensor_msgs::msg::JointState>(
    kJointStateTopic, measurement_qos(),
    [this](sensor_msgs::msg::JointState::ConstSharedPtr message) {
      on_joint_state(std::move(message));
    });
  payload_subscription_ = create_subscription<crane_msgs::msg::PayloadEstimate>(
    kPayloadEstimateTopic, payload_estimate_qos(),
    [this](crane_msgs::msg::PayloadEstimate::ConstSharedPtr message) {
      on_payload_estimate(std::move(message));
    });
  payload_service_ = create_service<crane_msgs::srv::SetPayload>(
    kSetPayloadService,
    [this](
      crane_msgs::srv::SetPayload::Request::SharedPtr request,
      crane_msgs::srv::SetPayload::Response::SharedPtr response) {
      on_set_payload(std::move(request), std::move(response));
    });

  timer_ = create_wall_timer(
    std::chrono::nanoseconds(static_cast<std::int64_t>(grid_.Ts * 1.0e9)), [this]() {update();});

  RCLCPP_INFO(
    get_logger(),
    "crane_mpc: %s at %.1f Hz, reliable and depth 1, in joint space. The horizon is %zu knots of "
    "%g s, so %.2f s of plan the receiver can execute without a further one -- "
    "control_architecture 3.3's fault-tolerance budget -- and it "
    "carries positions and velocities only (the design documentation). It is the output of the optimal control "
    "problem of wiki/mpc.md, solved with acados under RTI -- one iteration per cycle, ERK4 over "
    "%zu shooting intervals -- from the measured state carried %g s forward under the command "
    "already applied, tracking %s. Constraints 1 to 5 of mpc 3 are native boxes; 6 and 7 are "
    "native nonlinear cylinder-force and shared-pump rows, softened with reported L1 slack. With "
    "no reference, no measured state, or a plan that ended more than %g s "
    "ago, this node publishes nothing and the receiver runs its expiry ramp. A solve that does not "
    "converge, or that runs past the %.1f ms budget of mpc 4, is not published as a horizon: what "
    "goes out is mpc 6's previous solution shifted by one step, and %s carries the verdict, the "
    "residuals and SupervisorStatus FAULT_SOLVER every %zu solves -- which nothing consumes yet, "
    "the supervisor merging it being . Each cycle is warm-started from the previous "
    "solution shifted one step and cold-started from a rollout holding u = 0 on the first cycle "
    "and after every silence (mpc 6). After %zu consecutive solves that do not converge this node "
    "**stops publishing** and raises the escalation on %s, handing control back rather than "
    "shifting a plan it no longer believes; it does not ramp anything, and the zero the machine "
    "then reaches is the receiver's own expiry ramp running out of plan.",
    kHorizonTopic, rate, grid_.horizon_length, grid_.Ts, grid_.duration(),
    ocp_settings_.horizon_length, sensor_to_valve_delay_, kReferenceTopic, max_reference_age_,
    1.0e3 * ocp_settings_.solve_budget_s, kSolverHealthTopic, health_decimation_,
    max_consecutive_failures_, kSolverHealthTopic);

  if (mode_ == Mode::Shadow) {
    RCLCPP_INFO(
      get_logger(),
      "This node is in **shadow** mode, which is the state the design documentation slice 6 ships and the one user "
      "story 68 asks for: the solutions are judged before they drive the machine. Nothing at all "
      "goes out on %s -- the publisher exists and is never called, because relying on the "
      "receiver's chained path to win would make safety a property of an activation order. "
      "Everything else runs at rate and the horizon that would have gone out is published on %s "
      "instead, so only the last hop is withheld. %s carries the judgement every cycle: the shadow "
      "command against the velocity the follower produced on %s, per axis at the same instant, "
      "beside the constraint activity and the solve's verdict. Reading it and deciding this MPC is "
      "good enough is a **human gate** (the design documentation) and nothing here scores it. The transition to "
      "active is not this node's to take: it is the `mode` parameter, the supervisor's mode "
      "arbitration is , and no profile sets it today.",
      kHorizonTopic, kShadowHorizonTopic, kShadowComparisonTopic, kControllerStateTopic);
  } else {
    RCLCPP_WARN(
      get_logger(),
      "This node is in **active** mode: what it solves goes out on %s, which crane_velocity_"
      "controller fits and executes whenever no trajectory controller is chained onto it. That is "
      "not the state either CBS profile ships -- the design documentation slice 6 is 'shadow -> active' and the "
      "transition is the supervisor's alone (ROS 2 Interfaces 4, 'One command path'), which is "
      " -- so a deployment that reached here set `mode` itself and owes the judgement "
      "the design documentation makes a human gate.",
      kHorizonTopic);
  }
}

void MpcNode::on_joint_state(sensor_msgs::msg::JointState::ConstSharedPtr message)
{
  if (!ready()) {
    return;
  }
  // The two broadcasters publish their halves in separate, partial messages, so each half is
  // matched by name and only its own arrival stamp moves. A message that names neither is not an
  // error -- it is a third publisher's joints.
  std::array<bool, crane_model::kActuatedDof> actuated_seen{};
  std::array<bool, crane_model::kPassiveDof> passive_seen{};
  for (std::size_t index = 0; index < message->name.size(); ++index) {
    if (index >= message->position.size() || index >= message->velocity.size()) {
      continue;
    }
    const double position = message->position[index];
    const double velocity = message->velocity[index];
    if (!std::isfinite(position) || !std::isfinite(velocity)) {
      continue;
    }
    for (std::size_t axis = 0; axis < crane_model::kActuatedDof; ++axis) {
      if (message->name[index] == joints_[axis]) {
        q_a_measured_[axis] = position;
        dq_a_measured_[axis] = velocity;
        actuated_seen[axis] = true;
      }
    }
    for (std::size_t row = 0; row < crane_model::kPassiveDof; ++row) {
      if (message->name[index] == passive_joints_[row]) {
        q_u_measured_[row] = position;
        dq_u_measured_[row] = velocity;
        passive_seen[row] = true;
      }
    }
  }

  const bool actuated_complete =
    std::all_of(actuated_seen.begin(), actuated_seen.end(), [](bool found) {return found;});
  const bool passive_complete =
    std::all_of(passive_seen.begin(), passive_seen.end(), [](bool found) {return found;});
  const auto stamp = rclcpp::Time(message->header.stamp, RCL_ROS_TIME);
  if (actuated_complete) {
    joint_state_stamp_ = stamp;
    joint_state_seen_ = true;
    ++measurements_;
  }
  if (passive_complete) {
    passive_stamp_ = stamp;
    passive_seen_ = true;
    ++passive_measurements_;
  }
  if (
    !actuated_complete && !passive_complete &&
    std::any_of(actuated_seen.begin(), actuated_seen.end(), [](bool found) {return found;}))
  {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), kWarnPeriodMs,
      "A message on %s named some of the six actuated joints but not all of them, so the actuated "
      "half was not stamped. There is no plan for a state that is partly known.",
      kJointStateTopic);
  }
}

void MpcNode::on_controller_state(
  control_msgs::msg::JointTrajectoryControllerState::ConstSharedPtr message)
{
  controller_state_ = std::move(message);
  ++follower_states_;
}

MpcNode::FollowerCommand MpcNode::follower_command(const rclcpp::Time & now) const
{
  FollowerCommand follower;
  if (controller_state_ == nullptr) {
    return follower;
  }
  const control_msgs::msg::JointTrajectoryControllerState & state = *controller_state_;
  follower.age = (now - rclcpp::Time(state.header.stamp, RCL_ROS_TIME)).seconds();
  if (-follower.age > max_clock_skew_) {
    follower.source = "future";
    return follower;
  }
  if (follower.age > max_state_age_) {
    follower.source = "stale";
    return follower;
  }

  const char * source = nullptr;
  bool mixed = false;
  bool complete = true;
  for (std::size_t axis = 0; axis < crane_model::kActuatedDof; ++axis) {
    const auto row = std::find(state.joint_names.begin(), state.joint_names.end(), joints_[axis]);
    if (row == state.joint_names.end()) {
      complete = false;
      continue;
    }
    const auto index = static_cast<std::size_t>(std::distance(state.joint_names.begin(), row));

    const char * from = nullptr;
    if (index < state.output.velocities.size() && std::isfinite(state.output.velocities[index])) {
      follower.velocity[axis] = state.output.velocities[index];
      from = "output.velocities";
    }
    if (from == nullptr && index < state.reference.velocities.size() &&
      std::isfinite(state.reference.velocities[index]))
    {
      follower.velocity[axis] = state.reference.velocities[index];
      from = "reference.velocities";
    }
    if (from == nullptr) {
      complete = false;
    } else {
      follower.have_velocity[axis] = true;
      if (source == nullptr) {
        source = from;
      } else if (std::string(source) != from) {
        mixed = true;
      }
    }

    if (index < state.error.velocities.size() && std::isfinite(state.error.velocities[index])) {
      follower.velocity_error[axis] = state.error.velocities[index];
      follower.have_velocity_error[axis] = true;
    }
  }

  follower.complete = complete;
  follower.source = mixed ? "mixed" : (source == nullptr ? "none" : source);
  return follower;
}

crane_model::Input MpcNode::follower_input(const FollowerCommand & follower) const
{
  crane_model::Input input = crane_model::Input::Zero();
  if (!follower.complete) {
    return input;
  }
  for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
    const auto entry = static_cast<Eigen::Index>(row);
    const double implied = (follower.velocity[row] - dq_a_measured_[row]) / grid_.Ts;
    const double bound = ocp_settings_.limits.u_max[row];
    input[entry] = std::max(-bound, std::min(bound, implied));
  }
  return input;
}

void MpcNode::adopt_mode(Mode requested)
{
  if (requested == mode_) {
    return;
  }
  const Mode previous = mode_;
  mode_ = requested;

  last_horizon_.clear();
  tcp_horizon_states_.clear();
  last_tcp_horizon_states_.clear();
  warm_start_ = InitialGuess{};
  consecutive_failures_ = 0;
  escalated_ = false;
  cadence_anchored_ = false;
  last_input_ = crane_model::Input::Zero();

  RCLCPP_INFO(
    get_logger(),
    "The mode moved from %s to %s. The next solve **cold starts** (mpc 6): the warm start, the "
    "horizon 6's fallback would shift, its failure count and the publication cadence are all "
    "dropped, because every one of them was computed while something else was driving. %s.",
    to_string(previous), to_string(mode_),
    mode_ == Mode::Active ?
    "What this node solves now goes out on /crane/mpc/horizon and drives the machine" :
    "Nothing further goes out on /crane/mpc/horizon; the receiver runs its expiry ramp, which is "
    "the defined behaviour for a producer that has stopped");
}

void MpcNode::on_payload_estimate(crane_msgs::msg::PayloadEstimate::ConstSharedPtr message)
{
  payload_estimate_ = std::move(message);
}

void MpcNode::on_set_payload(
  crane_msgs::srv::SetPayload::Request::SharedPtr request,
  crane_msgs::srv::SetPayload::Response::SharedPtr response)
{
  response->active_mass = payload_.mass_kg;

  if (!ready()) {
    response->success = false;
    response->message = std::string("the MPC is not configured -- no robot description on ") +
      kRobotDescriptionTopic +
      " yet -- so there is no optimal control problem to set a payload on. Nothing is being "
      "published either, so nothing is being solved with the wrong load; declare it again once "
      "this node is up";
    RCLCPP_WARN(
      get_logger(), "%s was called before configuration and refused.", kSetPayloadService);
    return;
  }

  crane_model::Payload payload;
  std::string why;
  if (!payload_from_message(request->payload, payload, why)) {
    response->success = false;
    response->message = why + ". The payload this node is solving with is unchanged";
    RCLCPP_WARN(get_logger(), "%s refused a payload: %s.", kSetPayloadService, why.c_str());
    return;
  }

  const crane_model::Status status = ocp_->set_payload(payload);
  if (!status.ok()) {
    response->success = false;
    response->message = status.message + ". The payload this node is solving with is unchanged";
    RCLCPP_WARN(
      get_logger(), "%s refused a payload: %s.", kSetPayloadService, status.message.c_str());
    return;
  }

  const bool changed = payload.mass_kg != payload_.mass_kg ||
    payload.center_of_mass_k8_m != payload_.center_of_mass_k8_m;
  payload_ = payload;
  response->success = true;
  response->active_mass = payload_.mass_kg;
  if (!changed) {
    response->message = "the payload is unchanged, so nothing was dropped and the next solve is "
      "still warm-started";
    return;
  }

  warm_start_ = InitialGuess{};
  last_horizon_.clear();
  tcp_horizon_states_.clear();
  last_tcp_horizon_states_.clear();

  response->message = "the payload is now " + std::to_string(payload_.mass_kg) + " kg at (" +
    std::to_string(payload_.center_of_mass_k8_m.x()) + ", " +
    std::to_string(payload_.center_of_mass_k8_m.y()) + ", " +
    std::to_string(payload_.center_of_mass_k8_m.z()) +
    ") m in K8, on every stage of the next horizon. The next solve cold starts, because a payload "
    "step is a model change and the plan that was warm was a plan for the old one";
  RCLCPP_INFO(
    get_logger(),
    "%s adopted a payload of %.3f kg at (%.3f, %.3f, %.3f) m in K8. It is an acados parameter, so "
    "it reaches every stage of the next solve without a reconfigure; the warm start of mpc 6, the "
    "horizon its fallback would shift are both dropped, because the dynamics are different from "
    "the next cycle on. The sway box needs nothing dropped: it is drawn around the measured "
    "passive coordinate, which a payload step does not invalidate.",
    kSetPayloadService, payload_.mass_kg, payload_.center_of_mass_k8_m.x(),
    payload_.center_of_mass_k8_m.y(), payload_.center_of_mass_k8_m.z());
}

bool MpcNode::measured_state(
  const rclcpp::Time & now, crane_model::State & x, std::string & why) const
{
  why.clear();
  if (!joint_state_seen_) {
    why = std::string("no measured state has arrived on ") + kJointStateTopic;
    return false;
  }
  if (!passive_seen_) {
    why = std::string("no passive state has arrived on ") + kJointStateTopic;
    return false;
  }
  const double joint_age = (now - joint_state_stamp_).seconds();
  const double passive_age = (now - passive_stamp_).seconds();
  // `sensor_msgs/JointState` carries no validity flag, so freshness is the whole test: a passive
  // half that stopped being published is the only failure the source can report.
  if (joint_age > max_state_age_ || passive_age > max_state_age_) {
    why = "the measured state is older than max_state_age";
    return false;
  }

  x.setZero();
  for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
    x[static_cast<Eigen::Index>(kStateActuatedPosition + row)] = q_a_measured_[row];
    x[static_cast<Eigen::Index>(kStateActuatedVelocity + row)] = dq_a_measured_[row];
  }
  for (std::size_t row = 0; row < crane_model::kPassiveDof; ++row) {
    x[static_cast<Eigen::Index>(kStatePassivePosition + row)] = q_u_measured_[row];
    x[static_cast<Eigen::Index>(kStatePassiveVelocity + row)] = dq_u_measured_[row];
  }
  return true;
}

void MpcNode::on_robot_description(std_msgs::msg::String::ConstSharedPtr message)
{
  if (robot_description_ != nullptr) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), kWarnPeriodMs,
      "A second robot description arrived on %s and was ignored. A description that changes under "
      "a running horizon producer is a different crane, and rebuilding silently would publish a "
      "plan for the old one under the joint names of the new one.",
      kRobotDescriptionTopic);
    return;
  }

  robot_description_ = std::move(message);
  configure_if_inputs_ready();
}

void MpcNode::configure_if_inputs_ready()
{
  if (ready() || robot_description_ == nullptr) {
    return;
  }

  payload_.valid = true;
  payload_.mass_kg = 0.0;
  payload_.center_of_mass_k8_m.setZero();
  payload_.inertia_k8_kg_m2.setZero();

  crane_model::ModelConfig config;
  config.robot_description_xml = robot_description_->data;
  config.tool = crane_model::Tool::Pzs100;
  auto result = crane_model::Model::create(config);
  if (!result.ok()) {
    RCLCPP_ERROR(
      get_logger(),
      "The robot description on %s does not describe this machine, so there is no set of joint "
      "names to publish a horizon under: %s",
      kRobotDescriptionTopic, result.status().message.c_str());
    configuration_failure_ = result.status().message;
    return;
  }

  auto ocp = Ocp::create(config, payload_, ocp_settings_);
  if (!ocp.ok()) {
    RCLCPP_ERROR(
      get_logger(),
      "The optimal control problem of wiki/mpc.md could not be built from the description on %s, "
      "so this node is not configured and publishes nothing on %s: %s. A SymbolicBackendFailure "
      "here means the description builds a symbolic graph that then answers 0/0 -- a singular M_uu "
      "-- and running on one would be worse than not running.",
      kRobotDescriptionTopic, kHorizonTopic, ocp.status().message.c_str());
    configuration_failure_ = ocp.status().message;
    return;
  }
  ocp_ = std::move(ocp).value();
  model_.emplace(std::move(result).value());
  configuration_failure_.clear();

  const std::array<std::string, crane_model::kGeneralizedDof> & names =
    model_->urdf_joint_names();
  for (std::size_t axis = 0; axis < crane_model::kActuatedDof; ++axis) {
    joints_[axis] = names[kActuatedCanonical[axis]];
  }
  for (std::size_t row = 0; row < crane_model::kPassiveDof; ++row) {
    passive_joints_[row] = names[kPassiveCanonical[row]];
  }

  RCLCPP_INFO(
    get_logger(),
    "Model and OCP built once from %s, with a declared empty gripper. The payload is an acados "
    "parameter and %s is what moves it, at a grasp and at a release; %s is recorded and does not "
    "decide it (%s). The horizon is published under %s, %s, %s, %s, %s and %s.",
    kRobotDescriptionTopic, kSetPayloadService, kPayloadEstimateTopic,
    payload_estimate_ == nullptr ?
    "nothing has arrived on it yet" :
    (payload_estimate_->valid ? "it reports a valid estimate" : "it reports valid == false"),
    joints_[0].c_str(), joints_[1].c_str(), joints_[2].c_str(),
    joints_[3].c_str(), joints_[4].c_str(), joints_[5].c_str());

  const std::array<CylinderForceLimit, crane_model::kActuatedDof> & force =
    ocp_->cylinder_force_max();
  RCLCPP_INFO(
    get_logger(),
    "mpc 3 constraint 6 holds |F_cyl| inside extend/retract [%.3g/%.3g, %.3g/%.3g, %.3g/%.3g, "
    "%.3g/%.3g, %.3g/%.3g, %.3g/%.3g] N, derived from a %.3g Pa relief setting that is not "
    "measured on this machine; constraint 7 holds the summed pump draw under %.4g m^3/s, which "
    "is 0.95 x a 2023 pre-retrofit figure that has not been re-verified. The sixth pair is the "
    "tool cylinder's and is derived rather than imposed: the OCP does not plan the tool axis, so "
    "neither constraint carries a row for it.",
    force[0].extend, force[0].retract, force[1].extend, force[1].retract,
    force[2].extend, force[2].retract, force[3].extend, force[3].retract,
    force[4].extend, force[4].retract, force[5].extend, force[5].retract,
    ocp_settings_.hydraulics.system_pressure_pa, ocp_->pump_flow_max());

  adopt_reference();
}

void MpcNode::on_reference(trajectory_msgs::msg::JointTrajectory::ConstSharedPtr message)
{
  reference_message_ = std::move(message);
  adopt_reference();
}

void MpcNode::adopt_reference()
{
  if (reference_message_ == nullptr || !ready()) {
    return;
  }

  std::vector<Knot> incoming;
  std::string why;
  if (!reference_from_message(*reference_message_, joints_, incoming, why)) {
    RCLCPP_WARN(
      get_logger(), "The reference on %s was refused and the previous one stands: %s.",
      kReferenceTopic, why.c_str());
    reference_message_.reset();
    return;
  }

  reference_ = std::move(incoming);
  reference_stamp_ = rclcpp::Time(reference_message_->header.stamp, RCL_ROS_TIME);
  reference_message_.reset();
}

void MpcNode::stay_silent_after_failure(const std::string & why)
{
  last_silence_ = why;
  last_horizon_.clear();
  tcp_horizon_states_.clear();
  last_tcp_horizon_states_.clear();
  warm_start_ = InitialGuess{};
}

void MpcNode::stay_silent(const std::string & why)
{
  stay_silent_after_failure(why);
  consecutive_failures_ = 0;
  escalated_ = false;
}

bool MpcNode::shift_previous_horizon()
{
  if (last_horizon_.size() != horizon_.size() ||
    last_tcp_horizon_states_.size() != horizon_.size() || horizon_.empty())
  {
    return false;
  }
  for (std::size_t index = 0; index + 1U < horizon_.size(); ++index) {
    horizon_[index].q_a_ref = last_horizon_[index + 1U].q_a_ref;
    horizon_[index].dq_a_ref = last_horizon_[index + 1U].dq_a_ref;
    tcp_horizon_states_[index] = last_tcp_horizon_states_[index + 1U];
  }
  horizon_.back().q_a_ref = last_horizon_.back().q_a_ref;
  horizon_.back().dq_a_ref = last_horizon_.back().dq_a_ref;
  tcp_horizon_states_.back() = last_tcp_horizon_states_.back();
  return true;
}

void MpcNode::publish_tcp_horizon()
{
  if (!model_.has_value() || horizon_.empty() || tcp_horizon_states_.size() != horizon_.size()) {
    return;
  }

  nav_msgs::msg::Path path;
  path.header.frame_id = kTcpHorizonFrame;
  path.header.stamp = next_first_knot_;
  path.poses.reserve(horizon_.size());

  for (std::size_t index = 0; index < horizon_.size(); ++index) {
    const crane_model::State & state = tcp_horizon_states_[index];
    crane_model::Q q = crane_model::Q::Zero();
    q[0] = state[static_cast<Eigen::Index>(kStateActuatedPosition)];
    q[1] = state[static_cast<Eigen::Index>(kStateActuatedPosition + 1U)];
    q[2] = state[static_cast<Eigen::Index>(kStateActuatedPosition + 2U)];
    q[3] = state[static_cast<Eigen::Index>(kStateActuatedPosition + 3U)];
    q[4] = state[static_cast<Eigen::Index>(kStatePassivePosition)];
    q[5] = state[static_cast<Eigen::Index>(kStatePassivePosition + 1U)];
    q[6] = state[static_cast<Eigen::Index>(kStateActuatedPosition + 4U)];
    q[7] = state[static_cast<Eigen::Index>(kStateActuatedPosition + 5U)];

    const auto pose = model_->forward_kinematics(
      q, crane_model::Frame::MountingBase, crane_model::Frame::Tcp);
    if (!pose.ok()) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), kWarnPeriodMs,
        "The active robot model could not compute the TCP pose for MPC horizon knot %zu, so no "
        "visualization was published on %s: %s",
        index, kTcpHorizonTopic, pose.status().message.c_str());
      return;
    }

    geometry_msgs::msg::PoseStamped stamped_pose;
    stamped_pose.header.frame_id = kTcpHorizonFrame;
    stamped_pose.header.stamp =
      next_first_knot_ + rclcpp::Duration::from_seconds(horizon_[index].t);
    stamped_pose.pose.position.x = pose.value().position_m.x();
    stamped_pose.pose.position.y = pose.value().position_m.y();
    stamped_pose.pose.position.z = pose.value().position_m.z();
    stamped_pose.pose.orientation.w = pose.value().orientation.w();
    stamped_pose.pose.orientation.x = pose.value().orientation.x();
    stamped_pose.pose.orientation.y = pose.value().orientation.y();
    stamped_pose.pose.orientation.z = pose.value().orientation.z();
    path.poses.push_back(std::move(stamped_pose));
  }

  tcp_horizon_publisher_->publish(path);
}

void MpcNode::report_solver_health(const OcpSolution * solution, const std::string & why)
{
  using crane_msgs::msg::SolverHealth;
  using crane_msgs::msg::SupervisorStatus;

  crane_msgs::msg::SolverHealth & health = health_message_;
  health = crane_msgs::msg::SolverHealth{};
  health.header.stamp = this->now();
  for (std::size_t axis = 0; axis < crane_model::kActuatedDof; ++axis) {
    health.joint_names[axis] = joints_[axis];
  }
  health.solve_budget = ocp_settings_.solve_budget_s;
  health.applied_previous_solution = applied_previous_solution_;
  health.message = std::string(to_string(mode_)) + ": " + why;

  if (solution == nullptr) {
    health.outcome = SolverHealth::SOLVE_UNKNOWN;
    health.fault = SupervisorStatus::FAULT_SOLVER;
  } else {
    switch (solution->outcome) {
      case SolveOutcome::Converged:
        health.outcome = SolverHealth::SOLVE_CONVERGED;
        break;
      case SolveOutcome::BudgetExceeded:
        health.outcome = SolverHealth::SOLVE_BUDGET_EXCEEDED;
        break;
      case SolveOutcome::Failed:
        health.outcome = SolverHealth::SOLVE_FAILED;
        break;
    }
    health.fault = solution->outcome == SolveOutcome::Converged ?
      SupervisorStatus::FAULT_NONE : SupervisorStatus::FAULT_SOLVER;
    health.status = solution->status;
    health.status_word = solution->status_word;
    health.iterations = solution->iterations;
    health.qp_status = solution->qp_status;
    health.qp_iterations = solution->qp_iterations;
    health.solve_time = solution->solve_time_s;
    health.used_slack = solution->used_slack;
    health.slack_penalty = solution->slack_penalty;
    health.constraint_violation[SolverHealth::CONSTRAINT_SWAY] = solution->violation.q_u;
    health.constraint_violation[SolverHealth::CONSTRAINT_SWAY_RATE] = solution->violation.dq_u;
    health.constraint_violation[SolverHealth::CONSTRAINT_CYLINDER_FORCE] =
      solution->violation.cylinder_force;
    health.constraint_violation[SolverHealth::CONSTRAINT_PUMP_FLOW] = solution->violation.pump_flow;
  }

  const bool on_cadence = ((solves_ - 1U) % health_decimation_) == 0U;

  const bool verdict_changed = !last_health_.has_value() ||
    last_health_->fault != health.fault || last_health_->outcome != health.outcome;
  if (!on_cadence && !verdict_changed) {
    return;
  }

  if (solution != nullptr && ocp_ != nullptr) {
    const auto terms = ocp_->cost_terms(*solution, stages_);
    if (terms.ok()) {
      health.cost_term[SolverHealth::COST_ACTUATED_POSITION] = terms.value().q_a;
      health.cost_term[SolverHealth::COST_ACTUATED_VELOCITY] = terms.value().dq_a;
      health.cost_term[SolverHealth::COST_SWAY] = terms.value().q_u;
      health.cost_term[SolverHealth::COST_SWAY_RATE] = terms.value().dq_u;
      health.cost_term[SolverHealth::COST_EFFORT] = terms.value().tau_a;
      health.cost_term[SolverHealth::COST_INPUT] = terms.value().u;
      health.cost_term[SolverHealth::COST_TERMINAL] = terms.value().terminal;
      health.cost_term[SolverHealth::COST_SLACK] = terms.value().slack;
    } else {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), kWarnPeriodMs,
        "The cost of mpc 2 could not be split term by term for %s, so its rows are zero on this "
        "report rather than wrong: %s",
        kSolverHealthTopic, terms.status().message.c_str());
    }
  }

  health_publisher_->publish(health);
  last_health_ = health;
  ++health_published_;
}

void MpcNode::publish_shadow_comparison(const std::string & verdict, const OcpSolution * solution)
{
  using diagnostic_msgs::msg::DiagnosticStatus;
  using crane_msgs::msg::SolverHealth;

  diagnostic_msgs::msg::DiagnosticArray & message = comparison_message_;
  message = diagnostic_msgs::msg::DiagnosticArray{};
  message.header.stamp = this->now();
  message.status.resize(1);
  DiagnosticStatus & status = message.status.front();
  status.name = "crane_mpc: the shadow solution against what drove";
  status.hardware_id = "pzs100";

  std::vector<diagnostic_msgs::msg::KeyValue> & values = status.values;
  values.reserve(16U + 4U * crane_model::kActuatedDof);
  put(values, "mode", to_string(mode_));
  put(values, "follower.topic", kControllerStateTopic);
  put(values, "follower.velocity_source", follower_.source);
  put(values, "follower.age", as_text(follower_.age));
  put(values, "follower.error_field", "error.velocities");

  const bool converged = solution != nullptr && solution->outcome == SolveOutcome::Converged;
  put(values, "solve.converged", converged ? "true" : "false");
  put(
    values, "solve.outcome",
    solution != nullptr ? to_string(solution->outcome) : "the OCP refused its arguments");
  put(values, "solve.applied_previous_solution", applied_previous_solution_ ? "true" : "false");
  put(values, "solve.escalated", escalated_ ? "true" : "false");
  if (solution != nullptr) {
    put(values, "solve.time", as_text(solution->solve_time_s));
    put(values, "solve.budget", as_text(ocp_settings_.solve_budget_s));

    put(values, "constraint.used_slack", solution->used_slack ? "true" : "false");
    put(values, "constraint.slack_penalty", as_text(solution->slack_penalty));
    put(values, "constraint.sway", as_text(solution->violation.q_u));
    put(values, "constraint.sway_rate", as_text(solution->violation.dq_u));
    put(values, "constraint.cylinder_force", as_text(solution->violation.cylinder_force));
    put(values, "constraint.pump_flow", as_text(solution->violation.pump_flow));
  }

  std::size_t compared = 0;
  double largest = 0.0;
  for (std::size_t axis = 0; axis < crane_model::kActuatedDof; ++axis) {
    const std::string & joint = joints_[axis];
    if (shadow_command_valid_) {
      put(values, joint + ".shadow_velocity", as_text(shadow_command_[axis]));
    }
    if (follower_.have_velocity[axis]) {
      put(values, joint + ".follower_velocity", as_text(follower_.velocity[axis]));
    }
    if (shadow_command_valid_ && follower_.have_velocity[axis]) {
      const double difference = shadow_command_[axis] - follower_.velocity[axis];
      put(values, joint + ".difference", as_text(difference));
      largest = std::max(largest, std::abs(difference));
      ++compared;
    }
    if (follower_.have_velocity_error[axis]) {
      put(values, joint + ".follower_velocity_error", as_text(follower_.velocity_error[axis]));
    }
  }
  put(values, "difference.axes_compared", std::to_string(compared));
  if (compared > 0U) {
    put(values, "difference.largest_absolute", as_text(largest));
  }

  if (compared == crane_model::kActuatedDof) {
    status.level = DiagnosticStatus::OK;
    status.message = "the shadow command and the follower's are both present on all six axes; " +
      verdict;
  } else {
    status.level = DiagnosticStatus::WARN;
    status.message = std::string(
      shadow_command_valid_ ?
      "this cycle produced a shadow command and the follower's velocity is missing or stale on at "
      "least one axis, so those axes carry no difference; " :
      "this cycle produced no shadow command at all, so there is nothing to compare; ") + verdict;
  }

  comparison_publisher_->publish(message);
  last_comparison_ = message;
  ++comparisons_published_;
}

void MpcNode::update()
{
  if (parameters_->listener.try_update_params(parameters_->values)) {
    ocp_settings_.solve_budget_s = parameters_->values.solve_budget;
    health_decimation_ = static_cast<std::size_t>(parameters_->values.solver_health_decimation);
    Mode requested = mode_;
    if (mode_from_string(parameters_->values.mode, requested)) {
      adopt_mode(requested);
    }
    if (ocp_ != nullptr) {
      ocp_->set_solve_budget(ocp_settings_.solve_budget_s);
    }
  }

  shadow_command_valid_ = false;

  if (!ready()) {
    if (robot_description_ == nullptr) {
      stay_silent("no robot description has arrived");
    } else {
      stay_silent(configuration_failure_.empty() ?
        "configuration is incomplete" : configuration_failure_);
    }
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), kWarnPeriodMs,
      "The MPC is not configured (%s), so nothing is published on %s. Configuration requires one "
      "latched message on %s and nothing else -- the payload is an acados parameter and arrives "
      "on %s -- and the receiver's expiry ramp is the defined behaviour.",
      last_silence_.c_str(), kHorizonTopic, kRobotDescriptionTopic, kSetPayloadService);
    return;
  }
  if (reference_.empty()) {
    stay_silent("no reference has arrived");
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), kWarnPeriodMs,
      "No reference has arrived on %s, so nothing is published on %s. A horizon of zeros would be "
      "a plan to stop where the crane is not, and a stale one is worse; the receiver ramps.",
      kReferenceTopic, kHorizonTopic);
    return;
  }

  const rclcpp::Time now = this->now();

  const double lead = (reference_stamp_ - now).seconds();
  if (lead > max_clock_skew_) {
    stay_silent("the reference is stamped in this node's future");
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), kWarnPeriodMs,
      "The reference on %s is stamped %.3f s ahead of this node's clock, which is more than the "
      "%.3f s of skew this deployment allows, so nothing is published on %s. Every stamp here is "
      "on this node's own ROS clock and the receiver locates the present inside the horizon by it "
      "(control_architecture 3.3), so a horizon placed in the wrong second is worse than none: the "
      "expiry ramp is defined behaviour and a misplaced plan is not.",
      kReferenceTopic, lead, max_clock_skew_, kHorizonTopic);
    return;
  }

  const rclcpp::Time plan_ends =
    reference_stamp_ + rclcpp::Duration::from_seconds(reference_.back().t);
  if ((now - plan_ends).seconds() > max_reference_age_) {
    stay_silent("the reference's plan ended longer ago than max_reference_age");
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), kWarnPeriodMs,
      "The plan on %s ended %.3f s ago, which is past the %.3f s this node keeps holding its goal, "
      "so nothing is published on %s. The receiver runs its expiry ramp, which is the defined "
      "behaviour for a producer that has stopped (control_architecture 3.3, the design documentation); a publisher "
      "that kept emitting a dead plan would defeat it.",
      kReferenceTopic, (now - plan_ends).seconds(), max_reference_age_, kHorizonTopic);
    return;
  }

  const rclcpp::Time anchor = now + rclcpp::Duration::from_seconds(sensor_to_valve_delay_);
  const rclcpp::Time ceiling =
    now + rclcpp::Duration::from_seconds(sensor_to_valve_delay_ + grid_.duration());
  if (!cadence_anchored_ || next_first_knot_ < now || next_first_knot_ > ceiling) {
    next_first_knot_ = anchor;
    cadence_anchored_ = true;
  }

  double offset = (next_first_knot_ - reference_stamp_).seconds();
  if (offset < reference_.front().t) {
    offset = reference_.front().t;
  }

  const ResampleRejection rejection = resample(reference_, offset, grid_, horizon_);
  if (rejection != ResampleRejection::None) {
    stay_silent(to_string(rejection));
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), kWarnPeriodMs,
      "The reference on %s could not be resampled onto the horizon grid and nothing was published "
      "on %s: %s.",
      kReferenceTopic, kHorizonTopic, to_string(rejection));
    return;
  }


  crane_model::State measured;
  std::string why;
  if (!measured_state(now, measured, why)) {
    stay_silent(why);
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), kWarnPeriodMs,
      "Nothing was published on %s because there is no state to solve from: %s. mpc 1 initialises "
      "the OCP from the *measured* state, and a horizon computed from an assumed one is a plan for "
      "a machine that is not there; the receiver's expiry ramp is the defined behaviour instead "
      "(control_architecture 3.3).",
      kHorizonTopic, why.c_str());
    return;
  }

  follower_ = mode_ == Mode::Shadow ? follower_command(now) : FollowerCommand{};

  if (mode_ == Mode::Shadow) {
    last_input_ = follower_input(follower_);
  }
  const auto propagated = ocp_->propagate(measured, last_input_, sensor_to_valve_delay_);
  if (!propagated.ok()) {
    stay_silent("the measured state could not be propagated to the instant the plan takes effect");
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), kWarnPeriodMs,
      "The measured state could not be carried %g s forward under the command already applied, so "
      "nothing was published on %s: %s",
      sensor_to_valve_delay_, kHorizonTopic, propagated.status().message.c_str());
    return;
  }

  // Where the tool hangs, read off the measured passive coordinate rather than
  // solved for. `Model::passive_equilibrium` at the reference pose is a 5x5
  // grid search over the passive range plus a Newton solve to a 1e-8 residual
  // -- measured at 22.6 ms median on this image -- and it was re-run whenever
  // the pose moved `0.01` rad. At `dq_a_max` every axis clears that inside one
  // 40 ms cycle (slew 0.032 rad, telescope 0.025, arm 0.014, boom 0.013), so a
  // moving crane paid 22.6 ms ahead of a 7 ms solve on this node's
  // single-threaded executor, while a standing one paid nothing -- which is
  // why the horizon only ever came apart once the machine was moving. The
  // solve also refuses outright near the range corners, where the passive
  // joints reach no hanging pose from inside their limits.
  //
  // The state estimate already carries the answer: the tool hangs where it is
  // measured to hang. What a measurement cannot do is separate the hanging
  // pose from a sway *about* it, so while the tool is swinging, constraint 3's
  // box travels with the swing instead of holding still around the rest
  // position. That is a real loosening of the box and it is the trade this
  // makes. It is bounded: `weights.q_u` is zero on every shipped profile, so
  // this number is a box centre here and not a cost reference, and what damps
  // the sway is the `dq_u` term, which references zero and wants no
  // equilibrium at all.
  //
  // A constant will not do instead. The hanging pose is independent of slew,
  // telescope, rotator, tool and payload, but it tracks boom and arm exactly
  // -- `q_eq[0] = pi/2 - q_boom - q_arm` to 1e-4 across the workspace, a span
  // of 2.76 rad over the boom range alone against a box half-width of 0.2. A
  // frozen centre is the `q_eq = 0` case, which costs 296 active rows and 46
  // QP iterations against 21 and 19. `q_eq[1]` *is* constant at pi/2.
  for (std::size_t row = 0; row < crane_model::kPassiveDof; ++row) {
    q_eq_[static_cast<Eigen::Index>(row)] =
      propagated.value()[static_cast<Eigen::Index>(kStatePassivePosition + row)];
  }

  stages_.resize(horizon_.size());
  for (std::size_t index = 0; index < horizon_.size(); ++index) {
    for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
      stages_[index].q_a_ref[row] = horizon_[index].q_a_ref[row];
      stages_[index].dq_a_ref[row] = horizon_[index].dq_a_ref[row];
    }
    stages_[index].q_eq[0] = q_eq_[0];
    stages_[index].q_eq[1] = q_eq_[1];
  }

  auto solution = ocp_->solve(propagated.value(), stages_, warm_start_);
  ++solves_;
  if (!solution.ok()) {
    applied_previous_solution_ = false;
    ++consecutive_failures_;
    report_solver_health(nullptr, solution.status().message);
    if (mode_ == Mode::Shadow) {
      publish_shadow_comparison(solution.status().message, nullptr);
    }
    stay_silent_after_failure("the optimal control problem did not return a usable horizon");
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), kWarnPeriodMs,
      "The OCP returned no usable horizon and nothing was published on %s: %s",
      kHorizonTopic, solution.status().message.c_str());
    return;
  }
  last_solution_ = std::move(solution).value();
  if (!last_solution_->warm_started) {
    ++cold_starts_;
  }

  warm_start_ = last_solution_->outcome == SolveOutcome::Failed ?
    InitialGuess{} : shifted(*last_solution_);

  const bool converged = last_solution_->outcome == SolveOutcome::Converged;
  if (converged) {
    consecutive_failures_ = 0;
    escalated_ = false;
  } else {
    ++consecutive_failures_;
  }

  const char * const destination = mode_ == Mode::Active ? kHorizonTopic : kShadowHorizonTopic;

  std::string verdict;
  if (converged) {
    tcp_horizon_states_ = last_solution_->states;
    for (std::size_t index = 0; index < horizon_.size(); ++index) {
      const crane_model::State & state = last_solution_->states[index];
      for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
        horizon_[index].q_a_ref[row] =
          state[static_cast<Eigen::Index>(kStateActuatedPosition + row)];
        horizon_[index].dq_a_ref[row] =
          state[static_cast<Eigen::Index>(kStateActuatedVelocity + row)];
      }
    }
    applied_previous_solution_ = false;
    verdict = std::string("the solve converged inside the budget and its horizon was published "
      "on ") + destination;
  } else if (consecutive_failures_ >= max_consecutive_failures_) {
    escalated_ = true;
    applied_previous_solution_ = false;
    verdict = "mpc 6's repeated-failure escalation: " + std::to_string(consecutive_failures_) +
      " consecutive solves did not converge against a ceiling of " +
      std::to_string(max_consecutive_failures_) + " (acados last answered " +
      last_solution_->status_word + ", " + to_string(last_solution_->outcome) +
      "), so this node has stopped publishing and handed control back rather than shifting a plan "
      "it no longer believes";
    report_solver_health(&*last_solution_, verdict);
    if (mode_ == Mode::Shadow) {
      publish_shadow_comparison(verdict, &*last_solution_);
    }
    stay_silent_after_failure("mpc 6's repeated-failure escalation has stopped the publisher");
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), kWarnPeriodMs,
      "%s. Nothing further goes out on %s until a solve converges again; %s carries "
      "SupervisorStatus FAULT_SOLVER, and the supervisor acting on it is . This node does "
      "not ramp: the velocity command reaches zero on the *receiver's* horizon_expiry_ramp running "
      "out of plan (control_architecture 3.3), which is a different mechanism from this one and is "
      "the only one that produces a zero here.",
      verdict.c_str(), kHorizonTopic, kSolverHealthTopic);
    return;
  } else if (shift_previous_horizon()) {
    applied_previous_solution_ = true;
    verdict = std::string("acados answered ") + last_solution_->status_word + " (" +
      to_string(last_solution_->outcome) + ") after " +
      std::to_string(static_cast<int>(1.0e3 * last_solution_->solve_time_s + 0.5)) +
      " ms against a " +
      std::to_string(static_cast<int>(1.0e3 * ocp_settings_.solve_budget_s + 0.5)) +
      " ms budget, so mpc 6's previous solution shifted by one step was published on " +
      destination + " instead";
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), kWarnPeriodMs,
      "%s. mpc 5.3 requirements 1 and 3: a solve that did not converge is not published as a "
      "horizon, and %s carries the verdict with SupervisorStatus FAULT_SOLVER.",
      verdict.c_str(), kSolverHealthTopic);
  } else {
    applied_previous_solution_ = false;
    verdict = std::string("acados answered ") + last_solution_->status_word + " (" +
      to_string(last_solution_->outcome) +
      ") and there was no previous solution to shift, so nothing was published";
    report_solver_health(&*last_solution_, verdict);
    if (mode_ == Mode::Shadow) {
      publish_shadow_comparison(verdict, &*last_solution_);
    }
    stay_silent_after_failure(
      "the solve did not converge and there is no previous solution to shift");
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), kWarnPeriodMs,
      "%s on %s. The receiver runs its expiry ramp, which is the defined behaviour for a producer "
      "that has stopped and is the receiver's own mechanism, not a second one here.",
      verdict.c_str(), kHorizonTopic);
    return;
  }

  horizon_to_message(horizon_, joints_, next_first_knot_, message_);
  publish_tcp_horizon();
  if (mode_ == Mode::Active) {
    horizon_publisher_->publish(message_);
    ++published_;
  } else {
    shadow_horizon_publisher_->publish(message_);
    ++shadow_published_;
    for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
      shadow_command_[row] = horizon_[1].dq_a_ref[row];
    }
    shadow_command_valid_ = true;
  }
  last_horizon_ = horizon_;
  last_tcp_horizon_states_ = tcp_horizon_states_;
  if (converged) {
    last_input_ = last_solution_->u0;
  } else {
    for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
      last_input_[static_cast<Eigen::Index>(row)] =
        (horizon_[1].dq_a_ref[row] - horizon_[0].dq_a_ref[row]) / grid_.Ts;
    }
  }
  next_first_knot_ = next_first_knot_ + rclcpp::Duration::from_seconds(grid_.Ts);
  last_silence_.clear();
  report_solver_health(&*last_solution_, verdict);
  if (mode_ == Mode::Shadow) {
    publish_shadow_comparison(verdict, &*last_solution_);
  }
}

}
