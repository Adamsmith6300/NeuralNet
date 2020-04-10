using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Numpy;

namespace NeuralNet2
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();

            //MNIST_loader.serializeData();

            var (training_data, validation_data, test_data) = MNIST_loader.load_data_wrapper();
            Network net = new Network(new int[] { 784, 30, 10 });
            net.SGD(training_data, 30, 10, 3.0, test_data);

        }

    }
}
