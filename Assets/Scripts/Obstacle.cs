using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using NeuralNetwork;
using UnityEngine.UI;


public class Obstacle : MonoBehaviour 
{
	[SerializeField]
	private int numIter = 10; //number of iterations for training

	[SerializeField]
	private const double minError = 0.01;

	private static NeuralNet NN;

	private static List<DataPair> DataSet;

	bool trained;
	int i = 0;

	Text jump, around;

	ColliderScript script;

	public GameObject[] obstacles;
	private GameObject prefab;
	private GameObject prevObstacle = null;

	Animator anim;

	void Start ()
	{
		NN = new NeuralNet (3, 4, 1); //3 input, 4 hidden-later, one output
		DataSet = new List<DataPair>();
		jump = GameObject.Find ("jump").GetComponent<Text> ();
		around = GameObject.Find ("around").GetComponent<Text> ();
		jump.color = new Color (jump.color.r, jump.color.g, jump.color.b, 0);
		around.color = new Color (around.color.r, around.color.g, around.color.b, 0);
		NextExample ();
	}

	void NextExample()
	{
		Destroy (prevObstacle, 1.0f);
		StartCoroutine (Wait ());
	}
	IEnumerator Wait(){
		yield return new WaitForSeconds (1.1f);
		prefab = obstacles[Random.Range(0,obstacles.Length)];
		GameObject obstacleToAvoid = (GameObject)Instantiate (prefab);
		prevObstacle = obstacleToAvoid;

		Vector3 size = obstacleToAvoid.GetComponent<BoxCollider>().size;
		double[] inV = { (double)size.x, (double)size.y, (double)size.z };

		if (trained) {
			double[] output = NN.Compute (inV);
			if (output [0] > 0.5) {
				jump.color = new Color (jump.color.r, jump.color.g, jump.color.b, 1);
				around.color = new Color (around.color.r, around.color.g, around.color.b, 0);
			} 
			else {
				jump.color = new Color (jump.color.r, jump.color.g, jump.color.b, 0);
				around.color = new Color (around.color.r, around.color.g, around.color.b, 1);			}
		}

	}


	public void Train(float val)
	{
		//get dimensions of box collider on obstacles
		Vector3 size = prefab.GetComponent<BoxCollider> ().size;
		double[] inV = { (double)size.x, (double)size.y, (double)size.z };
		double[] dOut = { (double)val };
		DataSet.Add (new DataPair (inV, dOut));

		i++;
		if (!trained && i % numIter == (numIter - 1)) 
		{
			NN.Train (DataSet, minError);
			trained = true;
		}

		NextExample ();
	}
		
}
