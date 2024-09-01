"""Test model utities like get objects at time."""
import torch
from nerfstudio.cameras.cameras import Cameras, CameraType
from scipy.spatial.transform import Rotation as R

from map4d.common.geometry import opencv_to_opengl, rotate_z
from map4d.model.util import c2w_to_local, get_objects_at_time, opengl_frustum_check


def test_get_objects_at_time():
    """Test get_objects_at_time function."""
    object_poses = torch.rand(2, 3, 4)
    object_ids = torch.randperm(3).unsqueeze(0).repeat(2, 1)
    object_ids = torch.cat([object_ids, object_ids], dim=0)
    object_ids[0, -1] = -1
    object_dims = torch.rand(2, 3, 3)
    object_times = torch.tensor([0, 1])
    object_seq_ids = torch.tensor([0, 0])
    object_class_ids = torch.zeros_like(object_ids)
    sequence = torch.tensor([0])
    for time in torch.linspace(0, 1, 10):
        ids, poses, _, _ = get_objects_at_time(
            object_poses, object_ids, object_dims, object_times, object_seq_ids, object_class_ids, time, sequence
        )
        if time <= 0.5:
            assert len(ids) == 2
        else:
            assert len(ids) == 3

        for id, pose, ref_id, pt1, pt2 in zip(ids, poses, object_ids[0], object_poses[0], object_poses[1]):
            if ref_id == -1:
                assert torch.allclose(pose, pt2)
                continue
            assert id == ref_id
            assert torch.allclose(pose, (pt1 * (1 - time) + pt2 * time))


def test_c2w_to_local():
    """Test c2w_to_local function.

    We want to make sure that the viewdirs derived from the local camera and object points
    are consistent with the world space viewdirs from the original camera and world space
    points.
    """
    c2w = torch.eye(4).unsqueeze(0)
    c2w[0, :3, :3] = torch.from_numpy(R.from_euler("z", 10.0, degrees=True).as_matrix()).float()
    c2w[0, :3, 3] = torch.tensor([2.0, 3.0, 1.0])

    cam = Cameras(
        camera_to_worlds=c2w[:, :3],
        fx=torch.ones(1),
        fy=torch.ones(1),
        cx=torch.ones(1),
        cy=torch.ones(1),
        width=256,
        height=256,
        camera_type=CameraType.PERSPECTIVE,
        times=torch.zeros((1, 1)),
    )
    obj_pose = torch.tensor([3.0, 5.0, 0.5, torch.pi / 4]).unsqueeze(0)
    obj_dim = torch.tensor([1.0, 2.0, 1.0]).max()[None, None]
    obj_means = torch.rand(2, 3) - 0.5
    local_c2w = c2w_to_local(cam.camera_to_worlds, obj_pose, obj_dim)

    # calculate world space points
    obj_means_world = obj_means * obj_dim[0, 0]
    obj_means_world = rotate_z(obj_means_world, obj_pose[0, 3]) + obj_pose[0, :3]

    viewdirs = obj_means_world - cam.camera_to_worlds[..., :3, 3]  # (N, 3)
    viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)

    local_viewdirs = obj_means - local_c2w[..., :3, 3]  # (N, 3)
    local_viewdirs = local_viewdirs / local_viewdirs.norm(dim=-1, keepdim=True)
    viewdirs2 = rotate_z(local_viewdirs, obj_pose[0, 3])

    assert torch.allclose(
        viewdirs, viewdirs2, atol=1e-6
    ), f"Local viewdirs are not consistent with world viewdirs: {local_viewdirs} != {viewdirs}"


def test_opengl_frustum_check():
    """Test openGL frustum check used for computing current objects inside view."""
    # Setup camera
    camera = Cameras(
        camera_to_worlds=torch.eye(4)[:3].unsqueeze(0) @ opencv_to_opengl,
        fx=1.0,
        fy=1.0,
        cx=0.5,
        cy=0.5,
        width=1,
        height=1,
    )
    # Test objects fully inside view frustum
    obj_poses = torch.tensor([[0.1, 0, 1, 0]], dtype=torch.float32)
    obj_dims = torch.tensor([[0.1, 0.1, 0.1]], dtype=torch.float32)
    view_mask = opengl_frustum_check(obj_poses, obj_dims, camera)
    assert torch.all(view_mask)
    # Test objects partially inside view frustum
    obj_poses = torch.tensor([[0.4, 0, 1, 0]], dtype=torch.float32)
    obj_dims = torch.tensor([[0.2, 0.2, 0.2]], dtype=torch.float32)
    view_mask = opengl_frustum_check(obj_poses, obj_dims, camera)
    assert torch.all(view_mask)
    # Test objects outside view frustum
    obj_poses = torch.tensor([[1, 0, 1, 0]], dtype=torch.float32)
    obj_dims = torch.tensor([[0.1, 0.1, 0.1]], dtype=torch.float32)
    view_mask = opengl_frustum_check(obj_poses, obj_dims, camera)
    assert not torch.any(view_mask)
    # Test objects at edge of view frustum
    obj_poses = torch.tensor([[0.5, 0, 1, 0]], dtype=torch.float32)
    obj_dims = torch.tensor([[0.1, 0.1, 0.1]], dtype=torch.float32)
    view_mask = opengl_frustum_check(obj_poses, obj_dims, camera)
    assert torch.all(view_mask)
    # Test no objects
    obj_poses = torch.tensor([], dtype=torch.float32).reshape(0, 4)
    obj_dims = torch.tensor([], dtype=torch.float32).reshape(0, 3)
    view_mask = opengl_frustum_check(obj_poses, obj_dims, camera)
    assert len(view_mask) == 0
