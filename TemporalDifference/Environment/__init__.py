from TemporalDifference.Environment.cliff_walking_env_TD import CliffWalkingEnvTD


def env_builder(ncol, nrow):
    env = CliffWalkingEnvTD(ncol, nrow)
    return env