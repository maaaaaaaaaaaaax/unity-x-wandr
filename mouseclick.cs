using System.Collections;
using System.Collections.Generic;
using System;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.UI;
using System.Threading;




public class MouseClick : MonoBehaviour
{
    private static string F(float value)
    {
        return value.ToString("R", CultureInfo.InvariantCulture);
    }

    private Vector3 mouseClickPosition;
    private SMPLXController smplController;

    void Start()
        {
        Debug.Log("This is a test");
        smplController = FindObjectOfType<SMPLXController>();

        if (smplController == null)
        {
            Debug.LogError("SMPLXController NOT FOUND in scene!");
        }
        else
        {
            Debug.Log("SMPLXController found: " + smplController.name);
        }
    }

    void Update()
    {
        if (Input.GetMouseButtonDown(0)) // Left click
        {
            Vector3 screenPos = Input.mousePosition; // Screen coordinates
            Debug.Log($"Mouse Clicked at: {screenPos}");

            if (Camera.main == null)
            {
                Debug.LogError("No Main Camera found! Make sure your camera is tagged as 'MainCamera'.");
                return;
            }

            Ray ray = Camera.main.ScreenPointToRay(screenPos);
            Vector3 worldPos = Vector3.zero;

            if (Physics.Raycast(ray, out RaycastHit hit))
            {
                worldPos = hit.point;
                Debug.Log($"Raycast hit at: {worldPos}");
            }
            else
            {
                Debug.LogWarning("Raycast did not hit anything!");
            }



            // string message = $"{screenPos.x},{screenPos.y},{worldPos.x},{worldPos.y},{worldPos.z}";
            string smplData = smplController.GetInitialPoseData();
            string message_Debug = $"MouseClick: X:{F(worldPos.x)}, Y:{F(worldPos.y)}, Z:{F(worldPos.z)}; Data: {smplData}";
            string message = $"MouseClick:{F(worldPos.x)},{F(worldPos.y)},{F(worldPos.z)};{smplData}";


            Debug.Log("Sent: " + message_Debug);
            TCPManager.Instance.SendMessageToPython(message);
        }

        /*
        // Receive Data from Python and Move Cube
        string receivedData = TCPManager.Instance.GetReceivedData();
        if (!string.IsNullOrEmpty(receivedData))
        {
                Debug.Log("Received data: " + receivedData);
        }
        */
    }
}
