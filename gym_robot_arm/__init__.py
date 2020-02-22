from gym.envs.registration import register

register(
    id='robot-arm-v0',
    entry_point='gym_robot_arm.envs:RobotArmEnvV0',
)

register(
    id='robot-arm-v1',
    entry_point='gym_robot_arm.envs:RobotArmEnvV1',
)