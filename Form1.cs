using System;
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using Numpy;

namespace NeuralNet2
{
    public partial class Form1 : Form
    {
        Random rand;
        bool penDown;
        ArrayList points;
        Graphics g;
        Bitmap predictionX;

        Network net;
        List<(NDarray, byte)> td;
        List<(NDarray, NDarray)> trd;

        int hidden_neurons = 30;
        int _epochs = 5;
        int mini_batches = 10;
        Decimal learning_rate = 3.0M;

        public Form1()
        {
            var (training_data, validation_data, test_data) = MNIST_loader.load_data_wrapper();
            this.trd = training_data;
            this.td = test_data;

            //new network
            //net = new Network(new int[] { 784, 30, 10 });
            //net.SGD(training_data, 2, 10, 3.0, this.td);
            //Debug.WriteLine("DONE TRAINING!");
            //net.WriteNetwork("../../../data/network.nn");


            var (weights, biases) = Network.ReadNetwork("../../../data/network.nn");
            net = new Network(weights, biases);


            InitializeComponent();

            rand = new Random();
            textBox1.ReadOnly = true;
            numericUpDown1.Minimum = 11;
            numericUpDown1.Maximum = 200;
            numericUpDown1.Value = hidden_neurons;

            numericUpDown2.Minimum = 1;
            numericUpDown2.Maximum = 100;
            numericUpDown2.Value = _epochs;

            numericUpDown3.Minimum = 1;
            numericUpDown3.Maximum = 20;
            numericUpDown3.Value = mini_batches;

            numericUpDown4.DecimalPlaces = 3;
            numericUpDown4.Minimum = 0.001M;
            numericUpDown4.Maximum = 1000;
            numericUpDown4.Value = learning_rate;


            points = new ArrayList();
            penDown = false;
            g = panel1.CreateGraphics();
            panel1.BackColor = Color.FromArgb(255, 122, 242, 255);

            pictureBox1.SizeMode = PictureBoxSizeMode.Zoom;
            predictionX = new Bitmap(28, 28);

            clearPredictionX();

        }

        private void panel1_MouseDown(object sender, MouseEventArgs e)
        {
            Point newPoint = new Point(e.X, e.Y);
            points.Add(newPoint);
            penDown = true;

        }

        private void panel1_MouseUp(object sender, MouseEventArgs e)
        {
            penDown = false;
        }

        private void panel1_MouseMove(object sender, MouseEventArgs e)
        {
            Point nextPoint = new Point(e.X, e.Y);
            Pen pencil = new Pen(Color.Black);

            if (penDown)
            {
                if(points.Count > 1)
                {
                    g.DrawLine(pencil, (Point)points[points.Count - 1], nextPoint);
                }
                points.Add(nextPoint);
            }
        }

        private void button1_Click(object sender, EventArgs e)
        {
            g.Clear(Color.FromArgb(255, 122, 242, 255));
            points = new ArrayList();
            clearPredictionX();
        }

        private void button2_Click(object sender, EventArgs e)
        {
            foreach(Point p in points)
            {
                //Debug.WriteLine((p.X/10)+","+(p.Y/10));
                int x = (p.X / 10);
                int y = (p.Y / 10);
                predictionX.SetPixel(x, y, Color.White);
            }
            pictureBox1.Image = predictionX;

            double[,] sample = new double[28, 28];
            for (int i = 0; i < 28; ++i)
            {

                //Debug.Write("{");
                for (int j = 0; j < 28; ++j)
                {
                    Color c = predictionX.GetPixel(j, i);
                    int r = c.R;
                    sample[i, j] = (r / 255.0);
                    //Debug.Write(string.Format("{0:0.00}", (r / 255.0)));
                    //Debug.Write(",");
                }
                //Debug.WriteLine("}");
            }
            formatDigit(sample);


            byte prediction = net.evaluateSample(np.array(sample).reshape(784, 1));
            Debug.WriteLine("Prediction:");
            Debug.WriteLine(prediction);
            textBox1.Text = prediction.ToString();
        }

        private void button3_Click(object sender, EventArgs e)
        {
            int passed = net.evaluate(this.td);
            double percent = ((double)passed / 10000) * 100;
            Debug.WriteLine("Passed Tests: " + percent + "%");
            this.Text = "Passed: " + percent.ToString() + "% of test data";
        }

        private void button4_Click(object sender, EventArgs e)
        {
            hidden_neurons = (int)numericUpDown1.Value;
            _epochs = (int)numericUpDown2.Value;
            mini_batches = (int)numericUpDown3.Value;
            learning_rate = numericUpDown4.Value;

            //this.SuspendLayout();
            net = new Network(new int[] { 784, hidden_neurons, 10 });
            net.SGD(trd, _epochs, mini_batches, (double)learning_rate, null);
            Debug.WriteLine("DONE TRAINING!");
            this.Text = "Network Created...Benchmark Network to see results.";
            //net.WriteNetwork("../../../data/network.nn");
            //this.ResumeLayout();

        }

        private void button5_Click(object sender, EventArgs e)
        {
            var (weights, biases) = Network.ReadNetwork("../../../data/network.nn");
            net = new Network(weights, biases);
            this.Text = "Network Loaded from file...Benchmark Network to see results.";
        }

        public void clearPredictionX()
        {
            for (int i = 0; i < 28; ++i)
            {
                for (int j = 0; j < 28; ++j)
                {
                    predictionX.SetPixel(j, i, Color.Black);
                }
            }
        }

        public void formatDigit(double[,] image)
        {
            for(int row = 0; row < 28; ++row)
            {
                for(int col = 0; col < 28; ++col)
                {
                    bool isHorizontal = checkH(image, row, col);
                    if (isHorizontal) addPointsV(image, row, col);

                    bool isVertical = checkV(image, row, col);
                    if (isVertical) addPointsH(image, row, col);

                    bool isDiagPos = checkDiagPos(digit, row, col);
                    if (isDiagPos) addPointsDiagNeg(digit, row, col);

                    bool isDiagNeg = checkDiagNeg(digit, row, col);
                    if (isDiagNeg) addPointsDiagPos(digit, row, col);

                }
            }
        }




        public bool checkH(double[,] vals, int row, int col)
        {
            if(col-1 >= 0)
            {
                if (vals[row, col-1] > 0.7)
                {
                    return true;
                }

            }
            if (col + 1 < 28)
            {
                if (vals[row, col + 1] > 0.7)
                {
                    return true;
                }
            }
            return false;
        }

        public bool checkV(double[,] vals, int row, int col)
        {
            if (row - 1 >= 0)
            {
                if (vals[row - 1, col] > 0.7)
                {
                    return true;
                }

            }
            if (row + 1 < 28)
            {
                if (vals[row + 1, col] > 0.7)
                {
                    return true;
                }
            }
            return false;
        }


        public void addPointsV(double[,] vals, int row, int col)
        {
            if (row - 1 >= 0)
            {
                if (vals[row - 1, col] < 0.7)
                {
                    vals[row - 1, col] = Math.Max(0.1,rand.NextDouble() - 0.3);
                }

            }
            if (row + 1 < 28)
            {
                if (vals[row + 1, col] < 0.7)
                {
                    vals[row + 1, col] = Math.Max(0.1, rand.NextDouble() - 0.3);
                }
            }
        }

        public void addPointsH(double[,] vals, int row, int col)
        {
            if (col - 1 >= 0)
            {
                if (vals[row, col - 1] < 0.7)
                {
                    vals[row, col - 1] = Math.Max(0.1, rand.NextDouble() - 0.3);
                }
            }
            if (col + 1 < 28)
            {
                if (vals[row, col + 1] < 0.7)
                {
                    vals[row, col + 1] = Math.Max(0.1, rand.NextDouble() - 0.3);
                }
            }
        }



        public bool checkDiagPos(double[,] vals, int row, int col)
        {
            if (col - 1 >= 0 && row + 1 < 28)
            {
                if (vals[row + 1, col - 1] > 0.7)
                {
                    return true;
                }

            }
            if (col + 1 < 28 && row - 1 >= 0)
            {
                if (vals[row - 1, col + 1] > 0.7)
                {
                    return true;
                }
            }
            return false;
        }

        public bool checkDiagNeg(double[,] vals, int row, int col)
        {
            if (row - 1 >= 0 && col - 1 >= 0)
            {
                if (vals[row - 1, col - 1] > 0.7)
                {
                    return true;
                }

            }
            if (col + 1 < 28 && row + 1 < 28)
            {
                if (vals[row + 1, col + 1] > 0.7)
                {
                    return true;
                }
            }
            return false;
        }

        public void addPointsDiagPos(double[,] vals, int row, int col)
        {
            if (col - 1 >= 0 && row + 1 < 28)
            {
                if (vals[row + 1, col - 1] < 0.7)
                {
                    vals[row + 1, col - 1] = Math.Max(0.1, rand.NextDouble() - 0.3);
                }
            }
            if (col + 1 < 28 && row - 1 >= 0)
            {
                if (vals[row - 1, col + 1] < 0.7)
                {
                    vals[row - 1, col + 1] = Math.Max(0.1, rand.NextDouble() - 0.3);
                }
            }
        }

        public void addPointsDiagNeg(double[,] vals, int row, int col)
        {
            if (row - 1 >= 0 && col - 1 >= 0)
            {
                if (vals[row - 1, col - 1] < 0.7)
                {
                    vals[row - 1, col - 1] = Math.Max(0.1, rand.NextDouble() - 0.3);
                }
            }
            if (col + 1 < 28 && row + 1 < 28)
            {
                if (vals[row + 1, col + 1] < 0.7)
                {
                    vals[row + 1, col + 1] = Math.Max(0.1, rand.NextDouble() - 0.3);
                }
            }
        }

        public void printDigit(double[,] image)
        {
            for (int row = 0; row < 28; ++row)
            {
                Debug.Write("{");
                for (int col = 0; col < 28; ++col)
                {
                    Debug.Write(string.Format("{0:0.00}", image[row, col]));
                    Debug.Write(",");
                }
                Debug.WriteLine("}");
            }
        }


        public static double[,] digit = {
                        { 0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00},
                        { 0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00},
                        { 0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00},
                        { 0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00},
                        { 0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00},
                        { 0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00},
                        { 0.00,0.00,0.00,0.00,0.00,0.00,0.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00},
                        { 0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,1.00,1.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00},
                        { 0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,1.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00},
                        { 0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,1.00,1.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00},
                        { 0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,1.00,1.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00},
                        { 0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,1.00,1.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00},
                        { 0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,1.00,1.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00},
                        { 0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,1.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00},
                        { 0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,1.00,1.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00},
                        { 0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,1.00,1.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00},
                        { 0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,1.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00},
                        { 0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,1.00,1.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00},
                        { 0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,1.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00},
                        { 0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00},
                        { 0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00},
                        { 0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00},
                        { 0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00},
                        { 0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00},
                        { 0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00},
                        { 0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00},
                        { 0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00},
                        { 0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00} };

    }
}
