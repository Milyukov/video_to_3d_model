from scene_objects import Box

class Scene:

    def __init__(self) -> None:
        """Initialize scene as a number of geometric promitives in 3D
        """
        self.objects = []
        self.objects.append(Box(0, 0.5, 0, 1.0, 1.0, 1.0, 0.0, 0.0, 45.0))
        self.objects.append(Box(0, 1.25, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0))
        self.objects.append(Box(1.5, 0.5, -1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 30.0))