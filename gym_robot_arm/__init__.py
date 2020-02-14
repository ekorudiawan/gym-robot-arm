from gym.envs.registration import register

register(
    id='robot-arm-v0',
    entry_point='gym_robot_arm.envs:RobotArmEnv',
)