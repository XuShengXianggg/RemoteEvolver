"""RemoteEvolver: execution, tool evolution, Skill learning and evaluation."""
def __getattr__(name):
    if name == 'Agent':
        from .core import Agent
        return Agent
    raise AttributeError(name)

__all__ = ['Agent']
