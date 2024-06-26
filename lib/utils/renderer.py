# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import math
import trimesh
import pyrender
import numpy as np
from pyrender.constants import RenderFlags
from lib.models.smpl import get_smpl_faces


class WeakPerspectiveCamera(pyrender.Camera):
    def __init__(self,
                 scale,
                 translation,
                 znear=pyrender.camera.DEFAULT_Z_NEAR,
                 zfar=None,
                 name=None):
        super(WeakPerspectiveCamera, self).__init__(
            znear=znear,
            zfar=zfar,
            name=name,
        )
        self.scale = scale
        self.translation = translation

    def get_projection_matrix(self, width=None, height=None):
        P = np.eye(4)
        P[0, 0] = self.scale[0]
        P[1, 1] = self.scale[1]
        P[0, 3] = self.translation[0] * self.scale[0]
        P[1, 3] = -self.translation[1] * self.scale[1]
        P[2, 2] = -1
        return P


class Renderer:
    def __init__(self, resolution=(224,224), orig_img=False, wireframe=False, renderOnWhite = False):
        self.resolution = resolution

        self.faces = get_smpl_faces()
        self.orig_img = orig_img
        self.wireframe = wireframe
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=self.resolution[0],
            viewport_height=self.resolution[1],
            point_size=1.0
        )

        # set the scene
        self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))

        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1)

        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, -1, 1]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [0, 1, 1]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [1, 1, 2]
        self.scene.add(light, pose=light_pose)

        # [VIBE-Object Start]
        self.cam_node = None
        self.camera_pose = np.eye(4)
        self.fov = 0.0
        self.object_nodes = []
        self.human_nodes = []
        # [VIBE-Object End]

        if renderOnWhite:
            self.whiteBackground = np.full((self.resolution[1], self.resolution[0], 3), 255, dtype=np.uint8)

    # Original VIBE function to render a human.
    def render(self, img, verts, cam, angle=None, axis=None, mesh_filename=None, color=[1.0, 1.0, 0.9]):
        mesh = trimesh.Trimesh(vertices=verts, faces=self.faces, process=False)

        Rx = trimesh.transformations.rotation_matrix(math.radians(180), [1, 0, 0])
        mesh.apply_transform(Rx)

        if mesh_filename is not None:
            mesh.export(mesh_filename)

        if angle and axis:
            R = trimesh.transformations.rotation_matrix(math.radians(angle), axis)
            mesh.apply_transform(R)

        sx, sy, tx, ty = cam

        camera = WeakPerspectiveCamera(
            scale=[sx, sy],
            translation=[tx, ty],
            zfar=1000.
        )

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.5,
            alphaMode='OPAQUE',
            baseColorFactor=(color[0], color[1], color[2], 1.0)
        )

        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        mesh_node = self.scene.add(mesh, 'mesh')

        self.camera_pose = np.eye(4)
        cam_node = self.scene.add(camera, pose=self.camera_pose)

        if self.wireframe:
            render_flags = RenderFlags.RGBA | RenderFlags.ALL_WIREFRAME
        else:
            render_flags = RenderFlags.RGBA

        rgb, _ = self.renderer.render(self.scene, flags=render_flags)
        valid_mask = (rgb[:, :, -1] > 0)[:, :, np.newaxis]
        output_img = rgb[:, :, :-1] * valid_mask + (1 - valid_mask) * img
        image = output_img.astype(np.uint8)

        self.scene.remove_node(mesh_node)
        self.scene.remove_node(cam_node)

        return image

    def push_weak_cam(self, cam):
        sx, sy, tx, ty = cam
        camera = WeakPerspectiveCamera(
            scale=[sx, sy],
            translation=[tx, ty],
            zfar=1000.
        )
        self.camera_pose = np.eye(4)
        self.cam_node = self.scene.add(camera, pose=self.camera_pose)

    def push_persp_cam(self, yfov, cam_pose=np.eye(4)):
        self.fov = yfov
        camera = pyrender.PerspectiveCamera(yfov, 0.1, 1000.0)
        self.camera_pose = cam_pose
        self.cam_node = self.scene.add(camera, pose=self.camera_pose)
        
    def push_human(self, verts, color=[1.0, 1.0, 0.9], translation=[0.0, 0.0, 0.0]):
        # Build mesh from vertices.
        mesh = trimesh.Trimesh(vertices=verts, faces=self.faces, process=False)

        Rx = trimesh.transformations.rotation_matrix(math.pi, [1, 0, 0]) # Harcode another rotation because the human model loaded is using a different coordinate system.
        T = trimesh.transformations.translation_matrix(translation)

        # Apply transformations.
        mesh.apply_transform(Rx)
        mesh.apply_transform(T)

        # Material
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.5,
            alphaMode='OPAQUE',
            baseColorFactor=(color[0], color[1], color[2], 1.0)
        )

        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
        self.human_nodes.append(self.scene.add(mesh))

    def push_obj(self,
                   mesh_file,
                   translation_offset = [0.0, 0.0, 0.0],
                   translation=[0.0, 0.0, 0.0],
                   angle=0.0, # Rotation Angle (Radians)
                   axis=[1.0, 0.0, 0.0], # Rotation Axis (Right-Hand System: X points right. Y points up. Z points out.)
                   scale=[1.0, 1.0, 1.0],
                   color=[0.3, 1.0, 0.3]):
        # Load mesh from file.
        mesh = trimesh.load(mesh_file)

        T_Offset = trimesh.transformations.translation_matrix(translation_offset)

        # Apply transformations.
        Sx = trimesh.transformations.scale_matrix(scale[0], origin=[0,0, 0.0, 0.0], direction=[1.0, 0.0, 0.0])
        Sy = trimesh.transformations.scale_matrix(scale[1], origin=[0,0, 0.0, 0.0], direction=[0.0, 1.0, 0.0])
        Sz = trimesh.transformations.scale_matrix(scale[2], origin=[0,0, 0.0, 0.0], direction=[0.0, 0.0, 1.0])
        
        # prevent divide by zero error when angle is 0
        if angle == 0:
            R = trimesh.transformations.identity_matrix()
        else:
            R = trimesh.transformations.rotation_matrix(angle, axis)

        T = trimesh.transformations.translation_matrix(translation)
        
        mesh.apply_transform(Sx)
        mesh.apply_transform(Sy)
        mesh.apply_transform(Sz)
        mesh.apply_transform(T_Offset)
        mesh.apply_transform(R)
        mesh.apply_transform(T)

        # Setup material.
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.5,
            alphaMode='OPAQUE',
            baseColorFactor=(color[0], color[1], color[2], 1.0)
        )

        # Attach material to mesh and add it to scene.
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
        self.object_nodes.append(self.scene.add(mesh))

    def pop_and_render(self, img = None):
        # Render triangles or wireframe.
        if self.wireframe:
            render_flags = RenderFlags.RGBA | RenderFlags.ALL_WIREFRAME
        else:
            render_flags = RenderFlags.RGBA

        # background will just be white
        if img is None:
            img = self.whiteBackground

        # Combine current rendered scene with input image.
        # Allows multiple objects to be rendered by combining their resultant output.
        rgb, _ = self.renderer.render(self.scene, flags=render_flags)
        valid_mask = (rgb[:, :, -1] > 0)[:, :, np.newaxis]
        output_img = rgb[:, :, :-1] * valid_mask + (1 - valid_mask) * img
        image = output_img.astype(np.uint8)

        # Remove nodes
        self.scene.remove_node(self.cam_node)
        for n in self.object_nodes:
            self.scene.remove_node(n)
        for n in self.human_nodes:
            self.scene.remove_node(n)
        
        self.cam_node = None
        self.object_nodes.clear()
        self.human_nodes.clear()

        return image
    
    def screenspace_to_worldspace(self, ss_pos_x, ss_pos_y, ws_pos_z = 1.0):
            # Map the screen coordinate to NDC, which is [-1, 1].
            aspect_ratio = self.resolution[0]/self.resolution[1]

            ndc_x = -(ss_pos_x / self.resolution[0] * 2.0 - 1.0)
            ndc_y = ss_pos_y / self.resolution[1] * 2.0 - 1.0

            # Convert from NDC to world coordinate.
            ws_pos_x = ndc_x * ws_pos_z * math.tan(0.5 * self.fov) * aspect_ratio
            ws_pos_y = ndc_y * ws_pos_z * math.tan(0.5 * self.fov)

            return [ws_pos_x, ws_pos_y, ws_pos_z]
    # [VIBE-Object End]
