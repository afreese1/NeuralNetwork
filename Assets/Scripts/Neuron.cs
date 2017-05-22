using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

namespace NeuralNetwork
{
	public static class Sigmoid
	{
		public static double Output(double x)
		{
			if (x < -45)
				return 0.0f;
			else if (x > 45)
				return 1.0f;
			else
				return 1.0f / (1.0f + Mathf.Exp ((float)-x));
		}

		public static double FirstDerivative(double x)
		{
			return x * (1 - x);
		}
	}

	public class DataPair
	{

		public double[] inputVector { get; set; }
		public double[] desiredOutVector { get; set; }

		public DataPair(double[] inputVals, double[] desiredOutVals)
		{
			inputVector = inputVals;
			desiredOutVector = desiredOutVals;
		}
	}

	public class Connection
	{
		private System.Random Random = new System.Random ();
		public Neuron prevLayerNeuron { get; set; }
		public Neuron nextLayerNeuron { get; set; }
		public double weight { get; set; }
		public double deltaW { get; set; }

		public Connection(Neuron prevNeuron, Neuron nextNeuron)
		{
			prevLayerNeuron = prevNeuron;
			nextLayerNeuron = nextNeuron;
			weight = 2.0f * Random.NextDouble () - 1.0f;
		}
	}

	public class Neuron
	{
		public double value { get; set; }
		public double bias { get; set; }
		public double deltaB { get; set; }
		public double gradient { get; set; }
		public List<Connection> InputConnections { get; set; }
		public List<Connection> OutputConnections { get; set; }

		private System.Random Random = new System.Random();

		public Neuron()
		{
			InputConnections = new List<Connection> ();
			OutputConnections = new List<Connection> ();
			bias = 2.0f * Random.NextDouble () - 1.0f;
		}
	


		public Neuron (IEnumerable<Neuron> prevLayerNeurons): this()
		{
			foreach (Neuron n in prevLayerNeurons)
			{
				Connection c = new Connection (n, this);
				n.OutputConnections.Add(c);
				InputConnections.Add(c);
			}
		}

		public double NeuronOutput()
		{
			value = 0.0f;
			foreach(Connection i in InputConnections)
			{
				value += (i.weight*i.prevLayerNeuron.value)+bias;
			}
			value = Sigmoid.Output(value);
			return value;
		}

		public double LocalError(double desOut)
		{
			return desOut - value;
		}

		public double LocalGradient (double? desOut = null)
		{
			if(desOut != null)
			{
				gradient = LocalError(desOut.Value)*Sigmoid.FirstDerivative(value);
			}
			else
			{
				gradient = 0.0f;
				foreach (Connection o in OutputConnections)
				{
					gradient = o.weight*o.nextLayerNeuron.gradient*Sigmoid.FirstDerivative(value);
				}
			}
			return gradient;
		}

		public void UpdateWeights(double learningRate, double momentum)
		{
			double oldDeltaW;
			foreach (Connection i in InputConnections)
			{
				oldDeltaW = i.deltaW;
				i.deltaW = learningRate*gradient*i.prevLayerNeuron.value;
				i.weight += i. deltaW + momentum*oldDeltaW;
			}
			bias += learningRate*gradient + momentum*deltaB;
		}
	}
}