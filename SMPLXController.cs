﻿using System.Collections.Generic;
using System.Globalization;
using System.Text;
using UnityEngine;

/// <summary>
/// Transfers the current Unity SMPL-X pose to WANDR and applies generated
/// WANDR frames directly to the Unity character.
/// </summary>
public class SMPLXController : MonoBehaviour
{
    public const int NUM_JOINTS = 55;

    // WANDR generates the pelvis/local root plus the 21 SMPL-X body joints.
    public const int GENERATED_JOINTS = 22;

    private Dictionary<string, Transform> _joints;

    private readonly string[] _bodyJointNames =
    {
        "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
        "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar",
        "right_collar", "head", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "jaw", "left_eye_smplhf", "right_eye_smplhf", "left_index1",
        "left_index2", "left_index3", "left_middle1", "left_middle2", "left_middle3", "left_pinky1",
        "left_pinky2", "left_pinky3", "left_ring1", "left_ring2", "left_ring3", "left_thumb1",
        "left_thumb2", "left_thumb3", "right_index1", "right_index2", "right_index3", "right_middle1",
        "right_middle2", "right_middle3", "right_pinky1", "right_pinky2", "right_pinky3", "right_ring1",
        "right_ring2", "right_ring3", "right_thumb1", "right_thumb2", "right_thumb3"
    };

    private static string F(float value)
    {
        return value.ToString("R", CultureInfo.InvariantCulture);
    }

    private void Awake()
    {
        InitializeJoints();
    }

    private void InitializeJoints()
    {
        _joints = new Dictionary<string, Transform>();

        foreach (Transform joint in GetComponentsInChildren<Transform>(true))
        {
            _joints[joint.name] = joint;
        }

        foreach (string jointName in _bodyJointNames)
        {
            if (!_joints.ContainsKey(jointName))
            {
                Debug.LogWarning($"Joint '{jointName}' not found in the SMPL-X hierarchy.");
            }
        }
    }

    /// <summary>
    /// Serializes the current Unity translation, root orientation, and all 55
    /// local SMPL-X joint rotations for the TCP request to WANDR.
    /// </summary>
    public string GetInitialPoseData()
    {
        Vector3 bodyTranslation = transform.position;
        Quaternion bodyOrientation = NormalizedOrIdentity(transform.rotation);

        StringBuilder message = new StringBuilder();
        message.Append($"Translation:{F(bodyTranslation.x)},{F(bodyTranslation.y)},{F(bodyTranslation.z)};");
        message.Append($"Orientation:{F(bodyOrientation.x)},{F(bodyOrientation.y)},{F(bodyOrientation.z)},{F(bodyOrientation.w)};");
        message.Append("Pose:");

        for (int i = 0; i < NUM_JOINTS; i++)
        {
            Quaternion jointRotation = Quaternion.identity;

            if (_joints.TryGetValue(_bodyJointNames[i], out Transform joint))
            {
                jointRotation = NormalizedOrIdentity(joint.localRotation);
            }

            message.Append($"{F(jointRotation.x)},{F(jointRotation.y)},{F(jointRotation.z)},{F(jointRotation.w)}");

            if (i < NUM_JOINTS - 1)
            {
                message.Append(",");
            }
        }

        return message.ToString();
    }

    private void Update()
    {
        if (TCPManager.Instance != null &&
            TCPManager.Instance.TryGetNextFrame(out string receivedData))
        {
            ApplyModifiedPose(receivedData);
        }
    }

    [System.Serializable]
    public class PoseFrame
    {
        public float[] body_transl;
        public float[] body_orient;
        public float[] body_pose;
    }

    /// <summary>
    /// Applies one generated WANDR frame directly, without smoothing, root
    /// redirection, rest-pose multiplication, foot locking, or IK.
    /// </summary>
    public void ApplyModifiedPose(string json)
    {
        PoseFrame frame = JsonUtility.FromJson<PoseFrame>(json);
        if (frame == null)
        {
            return;
        }

        if (frame.body_transl != null && frame.body_transl.Length == 3)
        {
            transform.position = new Vector3(
                frame.body_transl[0],
                frame.body_transl[1],
                frame.body_transl[2]
            );
        }

        if (frame.body_orient != null && frame.body_orient.Length == 4)
        {
            transform.rotation = NormalizedOrIdentity(new Quaternion(
                frame.body_orient[0],
                frame.body_orient[1],
                frame.body_orient[2],
                frame.body_orient[3]
            ));
        }

        if (frame.body_pose == null || frame.body_pose.Length < GENERATED_JOINTS * 4)
        {
            return;
        }

        int jointCount = Mathf.Min(
            GENERATED_JOINTS,
            frame.body_pose.Length / 4
        );

        for (int i = 0; i < jointCount; i++)
        {
            if (!_joints.TryGetValue(_bodyJointNames[i], out Transform joint))
            {
                continue;
            }

            int offset = i * 4;
            joint.localRotation = NormalizedOrIdentity(new Quaternion(
                frame.body_pose[offset],
                frame.body_pose[offset + 1],
                frame.body_pose[offset + 2],
                frame.body_pose[offset + 3]
            ));
        }
    }

    private static Quaternion NormalizedOrIdentity(Quaternion quaternion)
    {
        float magnitude = Mathf.Sqrt(
            quaternion.x * quaternion.x +
            quaternion.y * quaternion.y +
            quaternion.z * quaternion.z +
            quaternion.w * quaternion.w
        );

        if (float.IsNaN(magnitude) ||
            float.IsInfinity(magnitude) ||
            magnitude < 0.000001f)
        {
            return Quaternion.identity;
        }

        float inverseMagnitude = 1f / magnitude;
        return new Quaternion(
            quaternion.x * inverseMagnitude,
            quaternion.y * inverseMagnitude,
            quaternion.z * inverseMagnitude,
            quaternion.w * inverseMagnitude
        );
    }
}
