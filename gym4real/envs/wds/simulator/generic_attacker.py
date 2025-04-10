class GenericAttacker:
    def __init__(self, name, target, event_start, event_end, tags=None):
        """

        """
        self.name = name
        self.target = target
        self.event_start = event_start
        self.event_end = event_end
        self.tags = tags

    def apply_attack(self, variables):
        pass


class MITM(GenericAttacker):
    def apply_attack(self, variables):
        """
        Inject the fake value for each tag declared in tags dictionary
        """
        for var in self.tags:
            variables[var['tag']][var['property']] = var['value']
        return variables


class DOS(GenericAttacker):

    def __init__(self, name, target, event_start, event_end, tags=None):
        super().__init__(name, target, event_start, event_end, tags)
        self.old_readings = {}

    def apply_attack(self, readings):
        """
        # TODO: starts the DOS from the second timestep, otherwise we need to get the previous env state
        """
        if not self.old_readings:
            self.old_readings = readings.copy()
        return self.old_readings.copy()


class NetworkDelay(GenericAttacker):
    def apply_attack(self, variables):
        return variables



