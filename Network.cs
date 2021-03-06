﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using Numpy;

namespace NeuralNet2
{
    class Network
    {
        public int[] sizes { get; set; }
        public int numLayers { get; set; }
        public NDarray biases { get; set; }
        public NDarray weights { get; set; }
        private static Random rng = new Random();


        public Network(List<List<List<double>>> newWeights, List<List<List<double>>> newBiases)
        {
            numLayers = 3;
            sizes = new int[] { newWeights[0][0].Count, newWeights[0].Count, newWeights[1].Count };

            List<NDarray> weights0 = new List<NDarray>();
            for (int i = 0; i < sizes[1]; ++i)
            {
                weights0.Add(np.array(newWeights[0][i].ToArray()));
            }

            List<NDarray> weights1 = new List<NDarray>();
            for (int i = 0; i < sizes[2]; ++i)
            {
                
                weights1.Add(np.array(newWeights[1][i].ToArray()));
            }
            List<NDarray> weights0And1 = new List<NDarray>();
            weights0And1.Add(np.array(weights0.ToArray()));
            weights0And1.Add(np.array(weights1.ToArray()));
            this.weights = new NDarray(np.array(weights0And1.ToArray()));

            List<NDarray> biases0 = new List<NDarray>();
            for (int i = 0; i < sizes[1]; ++i)
            {
                biases0.Add(np.array(newBiases[0][i].ToArray()));
            }

            List<NDarray> biases1 = new List<NDarray>();
            for (int i = 0; i < sizes[2]; ++i)
            {

                biases1.Add(np.array(newBiases[1][i].ToArray()));
            }
            List<NDarray> biases0And1 = new List<NDarray>();
            biases0And1.Add(np.array(biases0.ToArray()));
            biases0And1.Add(np.array(biases1.ToArray()));
            this.biases = new NDarray(np.array(biases0And1.ToArray()));

            Debug.WriteLine("Weights Shape:");
            Debug.WriteLine(this.weights.shape);
            Debug.WriteLine(this.weights[0].shape);
            Debug.WriteLine(this.weights[1].shape);

            Debug.WriteLine("Biases Shape:");
            Debug.WriteLine(this.biases.shape);
            Debug.WriteLine(this.biases[0].shape);
            Debug.WriteLine(this.biases[1].shape);

        }
        public Network(int[] sizes)
        {
            this.numLayers = sizes.Length;
            this.sizes = sizes;
            List<NDarray> newBiases = new List<NDarray>();
            for (int i = 1; i < sizes.Length; ++i)
            {
                newBiases.Add(np.random.randn(sizes[i], 1));
            }
            this.biases = np.array(newBiases.ToArray());
            List<NDarray> newWeights = new List<NDarray>();
            for (int i = 0; i < sizes.Length - 2; ++i)
            {
                newWeights.Add(np.random.randn(sizes[i + 1],sizes[i]));
            }
            for (int j = sizes.Length - 1; j > 1; --j)
            {
                newWeights.Add(np.random.randn(sizes[j], sizes[j - 1]));
            }
            this.weights = np.array(newWeights.ToArray());
        }

        public void SGD(List<(NDarray,NDarray)> training_data, int epochs, int mini_batch_size,
            double eta, List<(NDarray, byte)> test_data = null)
        {
            //if test_data: n_test = len(test_data)
            int n_test = 0;
            if (test_data != null)
            {
                n_test = test_data.Count;
            }
            //n = len(training_data)
            int n = training_data.Count;
            //for j in xrange(epochs):
            for(int i = 0; i < epochs; ++i)
            {
            //    random.shuffle(training_data)
                Shuffle(training_data);
                //    mini_batches = [training_data[k: k + mini_batch_size]
                //        for k in xrange(0, n, mini_batch_size)]
                List<List<(NDarray, NDarray)>> mini_batches = new List<List<(NDarray, NDarray)>>();
                for(int k = 0; k < n; k+= mini_batch_size)
                {
                    mini_batches.Add(training_data.GetRange(k, mini_batch_size));
                }
                //    for mini_batch in mini_batches:
                //        self.update_mini_batch(mini_batch, eta)
                foreach (var mini_batch in mini_batches)
                {
                    update_mini_batch(mini_batch, eta);
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                }
                //if test_data:
                //        print "Epoch {0}: {1} / {2}".format(
                //            j, self.evaluate(test_data), n_test)
                //    else:
                //        print "Epoch {0} complete".format(j)
                if (test_data != null)
                {
                    int passed_test = evaluate(test_data);
                    double percent = ((double)passed_test / n_test)*100;
                    Debug.WriteLine("Testing Epoch..." + percent +"%");
                    //if(percent > 93.0)
                    //{
                    //    SerializeNetwork(this, "../../../data/network.nn");
                    //    test_data = null;
                    //}
                } else
                {
                    Debug.WriteLine("Epoch " + (i+1) + " complete.");
                }
            }

        }

        public void Shuffle(List<(NDarray, NDarray)> list)
        {
            int n = list.Count;
            while (n > 1)
            {
                n--;
                int k = rng.Next(n + 1);
                var value = list[k];
                list[k] = list[n];
                list[n] = value;
            }
        }

        public void update_mini_batch(List<(NDarray, NDarray)> mini_batch, double eta)
        {
            //nabla_b = [np.zeros(b.shape) for b in self.biases]
            List<NDarray> nabla_b_ls = new List<NDarray>();
            for (int i = 0; i < biases.size; ++i)
            {
                nabla_b_ls.Add(np.zeros(biases[i].shape));
            }
            NDarray nabla_b = np.array(nabla_b_ls.ToArray());
            //nabla_w = [np.zeros(w.shape) for w in self.weights]
            List<NDarray> nabla_w_ls = new List<NDarray>();
            for (int i = 0; i < weights.size; ++i)
            {
                nabla_w_ls.Add(np.zeros(weights[i].shape));
            }
            NDarray nabla_w = np.array(nabla_w_ls.ToArray());
            foreach(var (x,y) in mini_batch)
            {
                //for x, y in mini_batch:
                //    delta_nabla_b, delta_nabla_w = self.backprop(x, y)
                //    nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                //      nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
                var (delta_nabla_b, delta_nabla_w) = backprop(x, y);
                var new_nabla_b = new List<NDarray>();

                for (int i = 0; i < nabla_b.size; ++i)
                {
                    new_nabla_b.Add(nabla_b[i] + delta_nabla_b[i]);
                }
                var new_nabla_w = new List<NDarray>();
                for (int i = 0; i < nabla_w.size; ++i)
                {
                    new_nabla_w.Add(nabla_w[i] + delta_nabla_w[i]);
                }
                nabla_b = np.array(new_nabla_b.ToArray());
                nabla_w = np.array(new_nabla_w.ToArray());
            }
            //    self.weights = [w - (eta / len(mini_batch)) * nw
            //                    for w, nw in zip(self.weights, nabla_w)]
            for (int j = 0; j < this.weights.size; ++j)
            {
                this.weights[j] = this.weights[j] - (eta / mini_batch.Count) * nabla_w[j];
            }
            //self.biases = [b - (eta / len(mini_batch)) * nb
            //               for b, nb in zip(self.biases, nabla_b)]
            for (int j = 0; j < this.biases.size; ++j)
            {
                this.biases[j] = this.biases[j] - (eta / mini_batch.Count) * nabla_b[j];
            }
        }

        public (NDarray, NDarray) backprop(NDarray x, NDarray y)
        {
            //nabla_b = [np.zeros(b.shape) for b in self.biases]
            //nabla_w = [np.zeros(w.shape) for w in self.weights]
            var new_nabla_b = new List<NDarray>();
            var new_nabla_w = new List<NDarray>();
            for (int j = 0; j < this.biases.size; ++j)
            {
                new_nabla_b.Add(np.zeros(this.biases[j].shape));
            }
            for (int j = 0; j < this.weights.size; ++j)
            {
                new_nabla_w.Add(np.zeros(this.weights[j].shape));
            }
            var nabla_b = np.array(new_nabla_b.ToArray());
            var nabla_w = np.array(new_nabla_w.ToArray());

            //# feedforward
            //activation = x
            var activation = x;
            //activations = [x] # list to store all the activations, layer by layer
            List<NDarray> activations = new List<NDarray>{ x };

            //zs = [] # list to store all the z vectors, layer by layer
            List<NDarray> zs = new List<NDarray>(); 
            //for b, w in zip(self.biases, self.weights):
            //    z = np.dot(w, activation) + b
            //    zs.append(z)
            //    activation = sigmoid(z)
            //    activations.append(activation)
            for(int i = 0; i < this.biases.size; ++i)
            {
                var b = this.biases[i];
                var w = this.weights[i];
                var z = np.dot(w, activation) + b;
                zs.Add(z);
                activation = sigmoid(z);
                activations.Add(activation);
            }
            //# backward pass
            //delta = self.cost_derivative(activations[-1], y) * \
            //    sigmoid_prime(zs[-1])
            var delta = cost_derivative(activations[activations.Count - 1], y) * sigmoid_prime(zs[zs.Count - 1]);
            
            nabla_b["-1"] = delta;
            var activTransp = activations[activations.Count - 2];
            nabla_w[nabla_w.size - 1] = np.dot(delta, np.transpose(activTransp));

            //# Note that the variable l in the loop below is used a little
            //# differently to the notation in Chapter 2 of the book.  Here,
            //# l = 1 means the last layer of neurons, l = 2 is the
            //# second-last layer, and so on.  It's a renumbering of the
            //# scheme in the book, used here to take advantage of the fact
            //# that Python can use negative indices in lists.
            //for l in xrange(2, self.num_layers):
            //    z = zs[-l]
            //    sp = sigmoid_prime(z)
            //    delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            //    nabla_b[-l] = delta
            //    nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
            //return (nabla_b, nabla_w)
            for (int i = 2; i < this.numLayers; ++i)
            {
                var z = zs[zs.Count - i];
                var sp = sigmoid_prime(z);
                delta = np.dot(np.transpose(this.weights[this.weights.size - 2 + 1]), delta) * sp;
                nabla_b[nabla_b.size - i] = delta;
                nabla_w[nabla_w.size - i] = np.dot(delta, np.transpose(activations[activations.Count - i - 1]));
            }

            return (nabla_b, nabla_w);
        }

        public int evaluate(List<(NDarray, byte)> test_data)
        {
            int count = 0;
            for (int i = 0; i < test_data.Count; ++i)
            {
                var (x, y) = test_data[i];
                var result = np.argmax(feedforward(x));
                var scalar = np.asscalar<byte>(result);

                if (scalar == y)
                {
                    count++;
                }
            }
            return count;
        }

        public byte evaluateSample(NDarray sample)
        {
            var result = np.argmax(feedforward(sample));
            var scalar = np.asscalar<byte>(result);
            return scalar;
        }

        public NDarray feedforward(NDarray a)
        {
            for (int i = 0; i < this.biases.size; ++i)
            {
                a = sigmoid(np.dot(this.weights[i], a) + this.biases[i]);
            }
            return a;
        }

        public NDarray sigmoid(NDarray z)
        {
            return 1.0 / (1.0 + np.exp(-z));
        }

        public NDarray sigmoid_prime(NDarray z)
        {
            return sigmoid(z) * (new double[] { 1.0 } - sigmoid(z));
        }

        /*
         * MAY BE ERRORS SWITCHING BETWEEN OVERLOAD METHODS
         */
        public double sigmoid(double z)
        {
            return 1.0 / (1.0 + Math.Exp(-z));
        }

        public double sigmoid_prime(double z)
        {
            return sigmoid(z) * (1.0 - sigmoid(z));
        }

        public NDarray cost_derivative(NDarray output_activations, NDarray y)
        {
            return (output_activations - y);
        }

        public void WriteNetwork(string fileName)
        {

            List<List<List<double>>> weights = new List<List<List<double>>>();
            List<List<double>> weights0 = new List<List<double>>();
            for (int i = 0; i < sizes[1]; ++i)
            {
                List<double> ls = new List<double>();
                for (int j = 0; j < sizes[0]; ++j)
                {
                    double w = (double)this.weights[0][i][j];
                    //Debug.WriteLine(w);
                    ls.Add(w);
                }
                weights0.Add(ls);
            }
            weights.Add(weights0);

            List<List<double>> weights1 = new List<List<double>>();
            for (int i = 0; i < sizes[2]; ++i)
            {
                List<double> ls = new List<double>();
                for (int j = 0; j < sizes[1]; ++j)
                {
                    double w = (double)this.weights[1][i][j];
                    //Debug.WriteLine(w);
                    ls.Add(w);
                }
                weights1.Add(ls);
            }
            weights.Add(weights1);

            List<List<List<double>>> biases = new List<List<List<double>>>();
            List<List<double>> biases0 = new List<List<double>>();
            for (int i = 0; i < sizes[1]; ++i)
            {
                List<double> ls = new List<double>();
                for (int j = 0; j < 1; ++j)
                {
                    double w = (double)this.biases[0][i][j];
                    //Debug.WriteLine(w);
                    ls.Add(w);
                }
                biases0.Add(ls);
            }
            biases.Add(biases0);

            List<List<double>> biases1 = new List<List<double>>();
            for (int i = 0; i < sizes[2]; ++i)
            {
                List<double> ls = new List<double>();
                for (int j = 0; j < 1; ++j)
                {
                    double w = (double)this.biases[1][i][j];
                    ls.Add(w);
                }
                biases1.Add(ls);
            }
            biases.Add(biases1);

            FileStream f =
                 new FileStream(fileName,
                 FileMode.Create);
            BinaryFormatter b = new BinaryFormatter();
            b.Serialize(f, (weights, biases));
            f.Close();

        }

        public static (List<List<List<double>>>, List<List<List<double>>>) ReadNetwork(string fileName)
        {
            FileStream f =
                 new FileStream(fileName,
                 FileMode.Open);
            BinaryFormatter b = new BinaryFormatter();
            (List<List<List<double>>>, List<List<List<double>>>) c = ((List<List<List<double>>>, List<List<List<double>>>))b.Deserialize(f);
            f.Close();
            return c;
        }

    }
}
