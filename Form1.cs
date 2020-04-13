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

        public Form1()
        {
            var (training_data, validation_data, test_data) = MNIST_loader.load_data_wrapper();
            this.td = test_data;
            //net = new Network(new int[] { 784, 30, 10 });
            //net.SGD(training_data, 2, 10, 3.0, this.td);
            //Debug.WriteLine("DONE TRAINING!");
            //net.WriteNetwork("../../../data/network.nn");

            var (weights, biases) = Network.ReadNetwork("../../../data/testNPWrite.nn");
            net = new Network(weights, biases);

            InitializeComponent();
            rand = new Random();
            textBox1.ReadOnly = true;
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
            //var (training_data, validation_data, test_data) = MNIST_loader.load_data_wrapper();
            
            int passed = net.evaluate(this.td);
            double percent = ((double)passed / 10000) * 100;
            Debug.WriteLine("Passed Tests: " + percent + "%");
            this.Text = "Passed: " + percent.ToString() + " of test data";
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


//MNIST_loader.serializeData();

//var (training_data, validation_data, test_data) = MNIST_loader.load_data_wrapper();
//Network net = new Network(new int[] { 784, 30, 10 });
//net.SGD(training_data, 1, 10, 3.0, test_data);


//Debug.WriteLine(net.weights[0].shape);//(2)(30,784)

//NDarray a = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });
//Debug.WriteLine(a.shape);

//a.tofile("../../../data/testNPWrite.txt", "", "%d");
//Debug.WriteLine(b.ToString());

//double[][][] wd = new double[net.weights.size][][];
//for (int i = 0; i < net.weights.size; ++i)
//{
//    wd[i] = new double[net.weights[i].size][];
//    for (int j = 0; j < net.weights[i].size; ++j)
//    {
//        wd[i][j] = new double[net.weights[i][j].size];
//        for(int k = 0; k < net.weights[i][j].size; ++k)
//        {
//            wd[i][j][k] = (double)net.weights[i][j][k];
//            //Debug.WriteLine(net.weights[i][j]);
//        }
//    }
//}
//var w = net.weights.GetData<double[][]>();
////Debug.WriteLine(w);
////Debug.WriteLine(w.Length);

//FileStream f =
//     new FileStream("../../../data/network.nn",
//     FileMode.Create);
//BinaryFormatter b = new BinaryFormatter();
//b.Serialize(f, (net.sizes, net.numLayers, w));
//f.Close();

//FileStream f2 =
//     new FileStream("../../../data/network.nn",
//     FileMode.Open);
//BinaryFormatter b2 = new BinaryFormatter();
//var (x, y, z) = ((int[], int, double[][]))b2.Deserialize(f2);
//f2.Close();

//Debug.WriteLine(np.array(z));