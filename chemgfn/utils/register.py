class ClassRegistry:
    def __init__(self):
        self._registry = {}

    def register(self, name):
        def decorator(cls):
            self._registry[name] = cls
            return cls

        return decorator

    def get_class(self, name):
        cls = self._registry.get(name)
        if cls is None:
            raise ValueError(f"Class not found for name: {name}")
        return cls
