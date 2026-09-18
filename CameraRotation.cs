using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraRotation : MonoBehaviour
{
    public float speed;
	//Vector3 pos;
    // Update is called once per frame
    void Update()
    {
        transform.Rotate(0, speed*Time.deltaTime, 0);
		//transform.Rotate(speed * Time.deltaTime, 0, 0);
		Debug.Log(transform.localEulerAngles.y);
	}
}
