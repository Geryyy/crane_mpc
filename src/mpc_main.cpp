// Executable entry point for MpcNode.
#include <memory>

#include "crane_mpc/mpc_node.hpp"
#include "rclcpp/rclcpp.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<crane_mpc::MpcNode>());
  rclcpp::shutdown();
  return 0;
}
