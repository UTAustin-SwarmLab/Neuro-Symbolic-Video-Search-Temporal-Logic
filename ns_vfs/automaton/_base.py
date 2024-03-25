import abc


class Automaton(abc.ABC):
    @abc.abstractclassmethod
    def set_up(self): ...
    @abc.abstractclassmethod
    def reset(self): ...
