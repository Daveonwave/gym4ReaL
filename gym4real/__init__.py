from gymnasium.envs.registration import register

register(
    id='gym4real/microgrid-v0',
    entry_point='gym4real.envs.microgrid.env:MicroGridEnv',
)

register(
    id='gym4real/wds_cps-v0',
    entry_point='gym4real.envs.wds.env_cps:WaterDistributionSystemEnv',
)

register(
    id='gym4real/wds-v0',
    entry_point='gym4real.envs.wds.env:WaterDistributionSystemEnv',
)

register(
    id='gym4real/robofeeder-v0',
    entry_point='gym4real.envs.robofeeder.Env_0:robotEnv',
)

register(
    id='gym4real/robofeeder-v1',
    entry_point='gym4real.envs.robofeeder.Env_1:robotEnv',
)

register(
    id='gym4real/robofeeder-v2',
    entry_point='gym4real.envs.robofeeder.Env_2:robotEnv',
)