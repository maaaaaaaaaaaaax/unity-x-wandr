using UnityEngine;

public class CameraFollow : MonoBehaviour
{
    [SerializeField] private Transform target;

    [Header("Fester Abstand in Welt-Richtung")]
    [SerializeField] private Vector3 offset = new Vector3(0f, 2f, -5f);

    [Header("Punkt, auf den die Kamera schaut")]
    [SerializeField] private Vector3 lookAtOffset = new Vector3(0f, 1f, 0f);

    private void LateUpdate()
    {
        if (target == null)
            return;

        // Kamera bewegt sich mit der Figur,
        // dreht ihre Position aber nicht mit der Figur.
        transform.position = target.position + offset;

        // Kamera schaut weiterhin auf die Figur.
        Vector3 lookTarget = target.position + lookAtOffset;
        transform.LookAt(lookTarget);
    }
}