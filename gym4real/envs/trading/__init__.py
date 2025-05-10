from gymnasium.envs.registration import register
from gym4real.envs.trading.env import TradingEnv

register(
    id="gym4real/TradingEnv-v0",
    entry_point=TradingEnv,
)