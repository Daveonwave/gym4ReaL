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