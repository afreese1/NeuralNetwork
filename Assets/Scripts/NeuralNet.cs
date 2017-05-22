using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

namespace NeuralNetwork
{
	public class NeuralNet
	{
		public List<Neuron> InputLayer { get; set; }
		public List<Neuron> HiddenLayer  { get; set; }
		public List<Neuron> OutputLayer { get; set; }

		public double learningRate { get; set; }
		public double momentum { get; set; }

		private static readonly System.Random Random = new System.Random();

		public NeuralNet(int numIn, int numHidden, int numOut, double? lr=null, double? m=null)
		{
			InputLayer = new List<Neuron> ();
			HiddenLayer = new List<Neuron> ();
			OutputLayer = new List<Neuron> ();

			int i;
			for(i=0; i<numIn; i++)
				InputLayer.Add(new Neuron());

			for(i=0; i < numHidden; i++)
				HiddenLayer.Add(new Neuron(InputLayer));
			
			for(i=0; i<numOut; i++)
				OutputLayer.Add(new Neuron(HiddenLayer));

			learningRate = lr ?? .1;
			momentum = m ?? .9;
		}

		public void Train(List<DataPair> DataPairs, double minError)
		{
			int numIter = 0; //number of iterations on
			double error = 1.0;

			while (error > minError && numIter < 10000000) {
				List<double> errors = new List<double> ();
				foreach (DataPair dp in DataPairs) {
					ForwardPropagate (dp.inputVector);
					BackPropagate (dp.desiredOutVector);
					errors.Add (OutputLayerError (dp.desiredOutVector));
				}
				error = errors.Average ();
				numIter++;
			}
		}

		private void ForwardPropagate(params double[] inputVector)
		{
			var i = 0;
			foreach (Neuron n in InputLayer) 
			{
				n.value = inputVector[i];
				i++;
			}

			foreach (Neuron n in HiddenLayer) 
			{
				n.value = n.NeuronOutput ();
			}

			foreach (Neuron n in OutputLayer)
			{
				n.value = n.NeuronOutput ();
			}
		}

		private void BackPropagate(params double[] desiredOutVector)
		{
			var i = 0;
			foreach (Neuron n in OutputLayer) 
			{
				n.LocalGradient (desiredOutVector [i]);
				i++;
			}

			foreach (Neuron n in HiddenLayer) 
			{
				n.LocalGradient ();
			}

			foreach (Neuron n in HiddenLayer) 
			{
				n.UpdateWeights (learningRate, momentum);
			}

			foreach (Neuron n in OutputLayer) 
			{
				n.UpdateWeights (learningRate, momentum);
			}

		}

		private double OutputLayerError(params double[] desiredOutVector)
		{
			var i = 0;
			double sum = 0.0f;
			foreach (Neuron n in OutputLayer) 
			{
				sum += Mathf.Abs ((float)n.LocalError (desiredOutVector [i]));
				i++;
			}
			return sum;
		}

		public double[] Compute (params double[] inputVector)
		{
			ForwardPropagate (inputVector);

			int i = 0;
			double[] outVector = new double[OutputLayer.Count];
			foreach(Neuron n in OutputLayer)
			{
				outVector[i] = n.value;
				i++;
			}
			return outVector;
		}
			
	}
}
