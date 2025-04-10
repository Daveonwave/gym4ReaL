from .env import WaterDistributionNetworkEnv

def run_no_agent_experiment(self, config_files):
        """
        Run a single no agent experiment.
        """
        env_config_path = Path(self.experiment_folder / config_files['env'])

        with open(env_config_path, 'r') as fin:
            env_settings = yaml.safe_load(fin)

        wn = WaterDistributionNetwork(env_settings['town'] + '.inp')
        wn.set_time_params(duration=env_settings['duration'], hydraulic_step=env_settings['hyd_step'])
        #TODO: set all the configuration needed before starting the experiment like tanks level and demand patterns
        wn.run()

        wn.create_df_reports()
        where = Path(self.experiment_folder / config_files['output_folder'])
        wn.save_csv_reports(where_to_save=where, save_links=False, save_nodes=False)
        
        
if __name__ == "__main__":
    import gym4real
    from pathlib import Path
    import yaml

    experiment_folder = Path(gym4real.__file__).parent.parent / 'experiments' / 'wds' / 'no_agent_experiment'
    config_files = {
        'env': 'env_config.yaml',
        'output_folder': 'output'
    }
    
    run_no_agent_experiment(experiment_folder, config_files)