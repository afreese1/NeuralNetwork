using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using NeuralNetwork;
using UnityStandardAssets.Characters.ThirdPerson;

public class ColliderScript : MonoBehaviour {

	[SerializeField]
	int ColliderType;

	public GameObject player;

	Obstacle script;

	Vector3 playerPosition;

	void Start()
	{
		playerPosition = player.transform.position;
		script = transform.parent.gameObject.GetComponent<Obstacle> ();
	}

	void OnTriggerEnter(Collider c)
	{
		if (c.tag == "Player") 
		{
			if (ColliderType == 0) {
				
				Debug.Log ("around");
				script.Train (0.0f); 
			} else {

				Debug.Log ("jump");
				script.Train (1.0f); 
			}
			StartCoroutine (ResetPlayer ());
			player.GetComponent<ThirdPersonUserControl> ().enabled = false;

		}
	}

	public IEnumerator ResetPlayer(){
		yield return new WaitForSeconds (1.0f);
		player.transform.position = playerPosition;
		player.GetComponent<ThirdPersonUserControl> ().enabled = true;
	}
}
