import random


def generate_single_attack_dict(name: str, attack_type: str, start: int, end: int, target: str, tags: list = None):
    return {
        'name': name,
        'type': attack_type,
        'trigger': {
            'type': 'time',
            'start': start,
            'end': end
        },
        'tags': tags,
        'target': target
    }


class AttackerGenerator:
    """
    Generates randomized attacks parsing the attackers' configuration file provided
    """
    TYPES = ['MITM', 'DOS']
    TARGETS = ['PLC2', 'PLC3']
    TAGS = {
        'PLC2': [{'tag': 'T41', 'property': ['pressure'], 'value': [0, 10.6]}],
        'PLC3': [{'tag': 'T42', 'property': ['pressure'], 'value': [0, 10.6]}]
    }
    MAX_OCCURRENCES = 5
    DURATIONS = [28800, 43200, 57600, 86400]
    SIMULATION_LENGTH = 604800
    PRESENCE_CHANCE = 0.8
    DOUBLE_ATTACK_CHANCE = 0.05

    def __init__(self,
                 attack_types: list = None,
                 targets: list = None,
                 tags_config: dict = None,
                 presence_chance: float = None,
                 max_occurrences: int = None,
                 episode_length: int = None,
                 durations: list = None,
                 double_attack_chance: float = None,
                 seed=0
                 ):
        self.attack_types = attack_types if attack_types is not None else self.TYPES
        self.targets = targets if targets is not None else self.TARGETS
        self.tags_config = tags_config if tags_config is not None else self.TAGS
        self.presence_chance = presence_chance if presence_chance is not None else self.PRESENCE_CHANCE
        self.max_occurrences = max_occurrences if max_occurrences is not None else self.MAX_OCCURRENCES
        self.episode_length = episode_length if episode_length is not None else self.SIMULATION_LENGTH
        self.durations = durations if durations is not None else self.DURATIONS
        self.double_attack_chance = double_attack_chance if double_attack_chance is not None else self.DOUBLE_ATTACK_CHANCE

        if seed:
            random.seed(seed)

    def create_attack_week(self):
        """
        Exposed method to create a list of weekly attacks
        """
        attack_list = self._build_weekly_attacks_list()
        return attack_list

    def _build_weekly_attacks_list(self):
        weekly_attacks = []

        presence_probability = random.uniform(0, 1)

        if presence_probability < self.presence_chance:
            n_attacks = random.randint(1, self.max_occurrences)

            last_timestep = 0
            for i in range(n_attacks):
                name = 'attack_' + str(i)

                # Sample attack type (unfair coin flip)
                # attack_type = random.choice(attack_types)
                coin_flip = random.uniform(0, 1)
                if coin_flip > 0.7:
                    attack_type = self.attack_types[1]
                else:
                    attack_type = self.attack_types[0]

                # Sample attack trigger
                duration = random.choice(self.durations)
                ith_lower_bound = (self.episode_length // n_attacks) * i
                lower_bound = ith_lower_bound if ith_lower_bound * i < last_timestep else last_timestep
                upper_bound = (self.episode_length // n_attacks) * i + (self.episode_length // n_attacks)
                start_time = random.randint(lower_bound, upper_bound - duration - 1)
                end_time = start_time + duration
                last_timestep = end_time

                # Double attack on both PLCs
                if random.uniform(0, 1) < self.double_attack_chance:
                    for target in self.targets:
                        tags = self._draw_attacked_tag_list(attack_type, target)
                        # Append the attack to the list of daily attacks
                        weekly_attacks.append(generate_single_attack_dict(name=name,
                                                                          attack_type=attack_type,
                                                                          start=start_time,
                                                                          end=end_time,
                                                                          target=target,
                                                                          tags=tags))
                # Single attack
                else:
                    # Sample attack target
                    target = random.choice(self.targets)
                    # Sample attacked tags based on type and target
                    tags = self._draw_attacked_tag_list(attack_type, target)
                    # Append the attack to the list of daily attacks
                    weekly_attacks.append(generate_single_attack_dict(name=name,
                                                                      attack_type=attack_type,
                                                                      start=start_time,
                                                                      end=end_time,
                                                                      target=target,
                                                                      tags=tags))
        return weekly_attacks

    # Choice of tag to involve in the selected attack based on attack_type and target
    def _draw_attacked_tag_list(self, attack_type, target):

        tags = []
        if attack_type == 'DOS':
            return None

        elif attack_type == 'MITM':
            # Get only one tag (modify to ask for multiple tags)
            n_sampels = 1
            sampled_tags = random.sample(self.tags_config[target], n_sampels)

            for tag in sampled_tags:
                var_name = tag['tag']
                prop = random.choice(tag['property'])
                value = random.choice(tag['value'])

                final_tag = {'tag': var_name, 'property': prop, 'value': value}
                tags.append(final_tag)

            return tags

        else:
            raise Exception("Wrong attack type!")
