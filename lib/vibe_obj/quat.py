import math
from .vec3 import Vec3

# Fucking mathutils.Quaternion's multiplication was fucking broken and returning me fucking garbage values!
# I wasted fucking hours of my life debugging that bullshit. Fuck that, I'll write my own fucking quaternion class.

class Quat:
    @classmethod
    def from_axis_angle(cls, _axis=Vec3(1.0, 0.0, 0.0), _angle=0.0):
        w = math.cos(_angle * 0.5)
        x = _axis.x * math.sin(_angle * 0.5)
        y = _axis.y * math.sin(_angle * 0.5)
        z = _axis.z * math.sin(_angle * 0.5)
        return cls(w, x, y, z)

    def __init__(self, _w, _x, _y, _z):
        self.w = _w
        self.x = _x
        self.y = _y
        self.z = _z

    def __mul__(self, _rhs):
        lhs_vec = Vec3(self.x, self.y, self.z)
        rhs_vec = Vec3(_rhs.x, _rhs.y, _rhs.z)
        w = self.w * _rhs.w - lhs_vec.dot(rhs_vec)
        xyz = self.w * rhs_vec + _rhs.w * lhs_vec + lhs_vec.cross(rhs_vec)
        return Quat(w, xyz.x, xyz.y, xyz.z)
    
    def __rmul__(self, _rhs):
        return _rhs.__mul__(self)

    def to_axis_angle(self):
        xyz = Vec3(self.x, self.y, self.z)
        if xyz.is_zero():
            axis = Vec3(1.0, 0.0, 0.0)
            angle = 0
            return (axis, angle)

        axis = xyz.normalised()
        angle = math.acos(self.w) * 2.0
        return (axis, angle)