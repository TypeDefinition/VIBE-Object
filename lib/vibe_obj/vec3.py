import math

class Vec3:
    def __init__(self, _x, _y, _z):
        self.x = _x
        self.y = _y
        self.z = _z

    def is_zero(self):
        return self.x == 0.0 and self.y == 0.0 and self.z == 0.0

    def dot(self, _rhs):
        return self.x * _rhs.x + self.y * _rhs.y + self.z * _rhs.z

    def cross(self, _rhs):
        x = self.y * _rhs.z - self.z * _rhs.y
        y = self.z * _rhs.x - self.x * _rhs.z
        z = self.x * _rhs.y - self.y * _rhs.x
        return Vec3(x, y, z)
    
    def length(self):
        return math.sqrt(self.length_squared())
    
    def length_squared(self):
        return self.x * self.x + self.y * self.y + self.z * self.z

    def normalised(self):
        length = self.length()
        return Vec3(self.x / length, self.y / length, self.z / length)

    def __mul__(self, _scalar):
        return Vec3(self.x * _scalar, self.y * _scalar, self.z * _scalar)
    
    __rmul__ = __mul__

    def __add__(self, _rhs):
        return Vec3(self.x + _rhs.x, self.y + _rhs.y, self.z + _rhs.z)
    
    def __sub__(self, _rhs):
        return Vec3(self.x - _rhs.x, self.y - _rhs.y, self.z - _rhs.z)