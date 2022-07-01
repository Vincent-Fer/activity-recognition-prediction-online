from .action import Action

class Activity:
    """
        This class is the representation of an activity.
        It contains informations about name of the activity and its state.
        It also contains informations about actions composing the activity (current action, ...)
    """

    def __init__(self, name, actions):
        """
            Initialiaze the activity object

            :param name:
            :type name: string
            :param actions: list of actions composing the activity
            :type actions: list of string
        """
        self.name = name
        self.complete = False
        self.create_action_list(actions)
        self.current_action_index = 0
        self.current_action = self.actions[0]

    def create_action_list(self, action_list):
        """
            Create action object for each action in the list of actions.

            :param action_list: list of actions composing the activity
            :type: list of string
        """

        self.actions = {}
        for action_index in range(len(action_list)):
            action_object = Action(action_list[action_index])
            self.actions[action_index] = action_object

    def go_to_next_action(self):
        """
            Set the state of the current action to True (Done)
            If activity not completed, set the next action in the list as current action 
            Else, set the state of the activity to True (Completed)
        """

        self.current_action.set_done(True)
        if (self.current_action_index + 1) in self.actions:
            self.current_action_index += 1
            self.current_action = self.actions[self.current_action_index]
        else:
            self.complete = True

    def is_complete(self):
        """
            Get the current state of the activity

            :return: current state
            :rtype: boolean
        """

        return self.complete

    def reset(self):
        """
            Reset the activity and actions composing it.
        """
        
        self.complete = False
        self.current_action_index = 0
        self.current_action = self.actions[self.current_action_index]
        for action in self.actions:
            self.actions[action].set_done(False)

    def __str__(self):
        res = ""
        res += "{}: ".format(self.name)
        for action in self.actions:
            res += self.actions[action].__str__()
            if action < len(self.actions.keys()) - 1:
                res += " -> "
        res += "\n"
        return res

    def print_web(self):
        res = ""
        res += "{}: ".format(self.name)
        for action in self.actions:
            res += self.actions[action].print_web()
            if action < len(self.actions.keys()) - 1:
                res += " -> "
        res += "\n"
        return res