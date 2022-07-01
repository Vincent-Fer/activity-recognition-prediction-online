class Action:
    """
        This class is the representation of an action
        It contains informations about the name and the current state of the action
    """

    def __init__(self, name):
        """
            Initialize the action object with a name and a state at False

            :param name: name of the action
            :type name: string
        """
    
        self.name = name
        self.done = False

    def get_name(self):
        """
            Used to get action's name

            :return: name of the action
            :rtype: string
        """
    
        return self.name
    
    def is_done(self):
        """
            Used to get action's state

            :return: state of the action
            :rtype: boolean
        """
    
        return self.done

    def set_done(self, done):
        """
            Set state of the action

            :param done: new state of the action
            :type done: boolean
        """
    
        self.done = done

    def print_web(self):
        res = ""
        if self.done: res += '<b style="color:blue">'
        res += self.name
        if self.done: res += '</b>'
        return res

    def __str__(self):
        res = ""
        if self.done: res += "\033[92m"
        res += self.name
        if self.done: res += "\033[0m"

        return res