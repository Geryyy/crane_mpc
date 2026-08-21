#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include "crane_model/testing/mock_model.hpp"

namespace crane_mpc
{
inline constexpr std::size_t kHorizonJointCount = 6;
using JointVector = std::array<double, kHorizonJointCount>;

struct MpcInput
{
  JointVector measured_position{};
  JointVector measured_velocity{};
  JointVector reference_position{};
  bool state_valid{false};
};

struct MpcOutput
{
  std::vector<JointVector> horizon;
  bool success{false};
  bool shadow_only{true};
  std::string message{};
};

class MpcContract
{
public:
  virtual ~MpcContract() = default;
  virtual MpcOutput solve(const MpcInput & input) const = 0;
};
}  // namespace crane_mpc

namespace
{
class MpcDouble final : public crane_mpc::MpcContract
{
public:
  crane_mpc::MpcOutput solve(const crane_mpc::MpcInput & input) const override
  {
    return {{input.reference_position, input.reference_position}, true, true, "fixture"};
  }
};
}  // namespace

TEST(CraneMpcContract, FixtureIsShadowOnlyAndFiniteLength)
{
  crane_mpc::MpcInput input;
  input.reference_position[2] = 4.0;
  const auto output = MpcDouble().solve(input);
  ASSERT_EQ(output.horizon.size(), 2U);
  EXPECT_TRUE(output.success);
  EXPECT_TRUE(output.shadow_only);
  EXPECT_DOUBLE_EQ(output.horizon[1][2], 4.0);
}

TEST(CraneMpcContract, W06UsesInstalledModelSymbolicGraphContract)
{
  const auto model = crane_model::testing::MockModel::create(crane_model::Tool::Pzs100);
  ASSERT_TRUE(model.ok());
  crane_model::Payload payload;
  payload.valid = true;
  payload.mass_kg = 1.0;
  const auto graph = model.value().symbolic_graph({}, payload);
  ASSERT_TRUE(graph.ok());
  EXPECT_EQ(graph.value().state_dimension(), crane_model::kStateDof);
  EXPECT_EQ(graph.value().input_dimension(), crane_model::kInputDof);
}
