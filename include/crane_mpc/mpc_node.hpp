
// ROS 2 node interface for solving and publishing MPC horizons.
#ifndef CRANE_MPC__MPC_NODE_HPP_
#define CRANE_MPC__MPC_NODE_HPP_

#include <array>
#include <chrono>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "control_msgs/msg/joint_trajectory_controller_state.hpp"
#include "crane_model/model.hpp"
#include "crane_mpc/horizon_source.hpp"
#include "crane_mpc/ocp_solver.hpp"
#include "crane_msgs/msg/payload_estimate.hpp"
#include "crane_msgs/msg/solver_health.hpp"
#include "crane_msgs/srv/set_payload.hpp"
#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/string.hpp"
#include "trajectory_msgs/msg/joint_trajectory.hpp"

namespace crane_mpc
{

inline constexpr char kReferenceTopic[] = "/crane/reference";

// The whole measured state arrives here, actuated and passive alike, from
// `joint_state_broadcaster`, off the interfaces the hardware component exports. It is still cached
// by joint name rather than by index, and still carries its own arrival stamp: a name-keyed cache
// costs nothing and does not depend on one producer publishing every joint in one message.
inline constexpr char kJointStateTopic[] = "/joint_states";

inline constexpr char kPayloadEstimateTopic[] = "/crane/payload_estimate";

inline constexpr char kControllerStateTopic[] = "/crane/controller_state";

inline constexpr char kRobotDescriptionTopic[] = "/robot_description";

inline constexpr char kSolverHealthTopic[] = "/crane/mpc/solver_health";

inline constexpr char kTcpHorizonTopic[] = "/crane/mpc/tcp_horizon";

inline constexpr char kTcpHorizonFrame[] = "K0_mounting_base";

inline constexpr char kSetPayloadService[] = "/crane/mpc/set_payload";

inline constexpr char kShadowHorizonTopic[] = "~/shadow_horizon";

inline constexpr char kShadowComparisonTopic[] = "~/shadow_comparison";

inline constexpr double kHorizonRate = 25.0;

[[nodiscard]] rclcpp::QoS horizon_qos();

[[nodiscard]] rclcpp::QoS reference_qos();

[[nodiscard]] rclcpp::QoS robot_description_qos();

[[nodiscard]] rclcpp::QoS measurement_qos();

[[nodiscard]] rclcpp::QoS payload_estimate_qos();

[[nodiscard]] rclcpp::QoS solver_health_qos();

[[nodiscard]] rclcpp::QoS controller_state_qos();

[[nodiscard]] rclcpp::QoS shadow_qos();

enum class Mode
{
  Shadow,
  Active,
};

[[nodiscard]] const char * to_string(Mode mode);

class MpcNode : public rclcpp::Node
{
public:
  explicit MpcNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~MpcNode() override;

    [[nodiscard]] bool ready() const noexcept {return model_.has_value() && ocp_ != nullptr;}

    void update();

  [[nodiscard]] const HorizonGrid & grid() const noexcept {return grid_;}

    [[nodiscard]] std::size_t published() const noexcept {return published_;}

  [[nodiscard]] Mode mode() const noexcept {return mode_;}

  [[nodiscard]] std::size_t shadow_published() const noexcept {return shadow_published_;}

    [[nodiscard]] std::size_t follower_states() const noexcept {return follower_states_;}

    [[nodiscard]] std::size_t comparisons_published() const noexcept
  {
    return comparisons_published_;
  }

  [[nodiscard]] const std::optional<diagnostic_msgs::msg::DiagnosticArray> &
  last_comparison() const noexcept
  {
    return last_comparison_;
  }

  [[nodiscard]] std::size_t reference_points() const noexcept {return reference_.size();}

    [[nodiscard]] const std::string & last_silence() const noexcept {return last_silence_;}

    [[nodiscard]] std::size_t measurements() const noexcept {return measurements_;}
  [[nodiscard]] std::size_t passive_measurements() const noexcept
  {
    return passive_measurements_;
  }

  [[nodiscard]] const std::array<std::string, crane_model::kActuatedDof> & joints() const noexcept
  {
    return joints_;
  }

  [[nodiscard]] const std::array<std::string, crane_model::kPassiveDof> &
  passive_joints() const noexcept
  {
    return passive_joints_;
  }

    [[nodiscard]] const std::optional<OcpSolution> & last_solution() const noexcept
  {
    return last_solution_;
  }

    [[nodiscard]] bool applied_previous_solution() const noexcept
  {
    return applied_previous_solution_;
  }

    [[nodiscard]] bool warm_started() const noexcept
  {
    return last_solution_.has_value() && last_solution_->warm_started;
  }

  [[nodiscard]] std::size_t cold_starts() const noexcept {return cold_starts_;}

  [[nodiscard]] std::size_t consecutive_failures() const noexcept
  {
    return consecutive_failures_;
  }

    [[nodiscard]] bool escalated() const noexcept {return escalated_;}

  [[nodiscard]] std::size_t max_consecutive_failures() const noexcept
  {
    return max_consecutive_failures_;
  }

  [[nodiscard]] const std::optional<crane_msgs::msg::SolverHealth> & last_health() const noexcept
  {
    return last_health_;
  }

  [[nodiscard]] std::size_t health_published() const noexcept {return health_published_;}

  [[nodiscard]] std::size_t solves() const noexcept {return solves_;}

  [[nodiscard]] const OcpSettings & ocp_settings() const noexcept {return ocp_settings_;}

  [[nodiscard]] double sensor_to_valve_delay() const noexcept {return sensor_to_valve_delay_;}

private:
    struct Parameters;

    struct FollowerCommand
  {
    std::array<double, crane_model::kActuatedDof> velocity{};
    std::array<bool, crane_model::kActuatedDof> have_velocity{};

    std::array<double, crane_model::kActuatedDof> velocity_error{};
    std::array<bool, crane_model::kActuatedDof> have_velocity_error{};

    const char * source{"none"};

    double age{0.0};

    bool complete{false};
  };

  void on_robot_description(std_msgs::msg::String::ConstSharedPtr message);
  void on_reference(trajectory_msgs::msg::JointTrajectory::ConstSharedPtr message);
  void on_joint_state(sensor_msgs::msg::JointState::ConstSharedPtr message);
  void on_payload_estimate(crane_msgs::msg::PayloadEstimate::ConstSharedPtr message);

    void on_set_payload(
    crane_msgs::srv::SetPayload::Request::SharedPtr request,
    crane_msgs::srv::SetPayload::Response::SharedPtr response);

  void on_controller_state(
    control_msgs::msg::JointTrajectoryControllerState::ConstSharedPtr message);

    [[nodiscard]] FollowerCommand follower_command(const rclcpp::Time & now) const;

    [[nodiscard]] crane_model::Input follower_input(const FollowerCommand & follower) const;

    void adopt_mode(Mode requested);

    void publish_shadow_comparison(const std::string & verdict, const OcpSolution * solution);

    void configure_if_inputs_ready();

    [[nodiscard]] bool measured_state(
    const rclcpp::Time & now, crane_model::State & x, std::string & why) const;

    void adopt_reference();

    void stay_silent_after_failure(const std::string & why);

    void stay_silent(const std::string & why);

    [[nodiscard]] bool shift_previous_horizon();

    void report_solver_health(const OcpSolution * solution, const std::string & why);

    void publish_tcp_horizon();

  std::unique_ptr<Parameters> parameters_;

  Mode mode_{Mode::Shadow};

  HorizonGrid grid_{};
  double max_reference_age_{0.5};
  double max_clock_skew_{0.04};
  double max_state_age_{0.2};
  double sensor_to_valve_delay_{0.05};

  OcpSettings ocp_settings_{};
  std::unique_ptr<Ocp> ocp_;
  std::optional<OcpSolution> last_solution_;

  std::size_t health_decimation_{2};
  std::size_t solves_{0};
  std::size_t health_published_{0};
  bool applied_previous_solution_{false};
  std::optional<crane_msgs::msg::SolverHealth> last_health_;
  crane_msgs::msg::SolverHealth health_message_;

    InitialGuess warm_start_{};
  std::size_t cold_starts_{0};

    std::size_t max_consecutive_failures_{5};
  std::size_t consecutive_failures_{0};
  bool escalated_{false};

  std::vector<Knot> last_horizon_;

  std::vector<crane_model::State> tcp_horizon_states_;
  std::vector<crane_model::State> last_tcp_horizon_states_;

  crane_model::Input last_input_{crane_model::Input::Zero()};

  std::optional<crane_model::Model> model_;
  std::array<std::string, crane_model::kActuatedDof> joints_{};
  std::array<std::string, crane_model::kPassiveDof> passive_joints_{};

  crane_model::QU q_eq_{crane_model::QU::Zero()};

  std::array<double, crane_model::kActuatedDof> q_a_measured_{};
  std::array<double, crane_model::kActuatedDof> dq_a_measured_{};
  rclcpp::Time joint_state_stamp_{0, 0, RCL_ROS_TIME};
  bool joint_state_seen_{false};
  std::size_t measurements_{0};

  std::array<double, crane_model::kPassiveDof> q_u_measured_{};
  std::array<double, crane_model::kPassiveDof> dq_u_measured_{};
  rclcpp::Time passive_stamp_{0, 0, RCL_ROS_TIME};
  bool passive_seen_{false};
  std::size_t passive_measurements_{0};

    crane_model::Payload payload_{};
  crane_msgs::msg::PayloadEstimate::ConstSharedPtr payload_estimate_;
  std_msgs::msg::String::ConstSharedPtr robot_description_;
  std::string configuration_failure_;

  std::vector<Knot> reference_;
  rclcpp::Time reference_stamp_{0, 0, RCL_ROS_TIME};

  /// How far into the reference the MPC has actually spent the plan, in seconds.
  /**
   * The horizon is sampled from the reference **here** and not at the wall-clock
   * offset. That is the whole point of the progress state: each converged solve
   * says how much of the plan its first interval consumes, this advances by
   * exactly that, and `s` restarts at zero -- Marc's spline-origin advance
   * (`timber_crane_mpc.cpp:173-181`). Indexing by wall clock instead is what let
   * the reference run away from a machine that had fallen behind, so that the
   * tracking cost pulled harder the further behind it got.
   *
   * It is re-anchored to the wall clock whenever a new reference arrives, or
   * after any cycle that published nothing: a plan whose origin is carried
   * across a silence is a plan for a machine nobody was commanding.
   */
  double reference_progress_{0.0};
  bool reference_progress_anchored_{false};

  trajectory_msgs::msg::JointTrajectory::ConstSharedPtr reference_message_;

  rclcpp::Time next_first_knot_{0, 0, RCL_ROS_TIME};
  bool cadence_anchored_{false};

  control_msgs::msg::JointTrajectoryControllerState::ConstSharedPtr controller_state_;
  std::size_t follower_states_{0};

    FollowerCommand follower_{};

    std::array<double, crane_model::kActuatedDof> shadow_command_{};
  bool shadow_command_valid_{false};

  std::vector<Knot> horizon_;
  trajectory_msgs::msg::JointTrajectory message_;
  diagnostic_msgs::msg::DiagnosticArray comparison_message_;
  std::optional<diagnostic_msgs::msg::DiagnosticArray> last_comparison_;
  std::size_t published_{0};
  std::size_t shadow_published_{0};
  std::size_t comparisons_published_{0};
  std::string last_silence_{};

  std::vector<OcpStage> stages_;

  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr horizon_publisher_;
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr shadow_horizon_publisher_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr tcp_horizon_publisher_;
  rclcpp::Publisher<crane_msgs::msg::SolverHealth>::SharedPtr health_publisher_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr comparison_publisher_;
  rclcpp::Subscription<control_msgs::msg::JointTrajectoryControllerState>::SharedPtr
    controller_state_subscription_;
  rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr reference_subscription_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr robot_description_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_subscription_;
  rclcpp::Subscription<crane_msgs::msg::PayloadEstimate>::SharedPtr payload_subscription_;
  rclcpp::Service<crane_msgs::srv::SetPayload>::SharedPtr payload_service_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}

#endif
