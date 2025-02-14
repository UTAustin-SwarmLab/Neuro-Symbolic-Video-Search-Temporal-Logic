import abc


class Automaton(abc.ABC):
    @abc.abstractmethod
    def set_up(self): ...
    @abc.abstractmethod
    def reset(self): ...
