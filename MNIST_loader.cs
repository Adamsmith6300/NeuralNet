using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading.Tasks;
using Numpy;

namespace NeuralNet2
{
    class MNIST_loader
    {

        public static (List<(NDarray, NDarray)>, List<(NDarray, byte)>, List<(NDarray, byte)>) load_data_wrapper()
        {
            List<(double[,], byte)> tr_d = DeSerializeNow("../../../trainingData.nn");
            //List<(double[,], byte)> tr_d = readData(60000, "../../../train-labels.idx1-ubyte", "../../../train-images.idx3-ubyte");
            List<(double[,], byte)> va_d = DeSerializeNow("../../../validationData.nn");
            //List<(double[,], byte)> va_d = tr_d.GetRange(50000, 10000);
            List<(double[,], byte)> te_d = DeSerializeNow("../../../testingData.nn");
            //List<(double[,], byte)> te_d = readData(10000, "../../../t10k-labels.idx1-ubyte", "../../../t10k-images.idx3-ubyte");

            List<(NDarray, NDarray)> training_data = new List<(NDarray, NDarray)>();
            foreach(var e in tr_d)
            {
                var (data, lbl) = e;
                NDarray data_nd = np.array(data).reshape(784,1);
                NDarray lbl_v = vectorized_result(lbl);
                training_data.Add((data_nd, lbl_v));
            }
            List<(NDarray, byte)> validation_data = new List<(NDarray, byte)>();
            foreach (var e in va_d)
            {
                var (data, lbl) = e;
                NDarray data_nd = np.array(data).reshape(784, 1);
                validation_data.Add((data_nd, lbl));
            }
            List<(NDarray, byte)> test_data = new List<(NDarray, byte)>();
            foreach (var e in te_d)
            {
                var (data, lbl) = e;
                NDarray data_nd = np.array(data).reshape(784, 1);
                test_data.Add((data_nd, lbl));
            }
            return (training_data,validation_data,test_data);
        }

        public static NDarray vectorized_result(byte j)
        {
            var e = np.zeros(new int[] { 10, 1 });
            e[j][0] = new int[1] { 1 };
            return e;
        }


        public static List<(double[,], byte)> readData(int size, string labelPath, string dataPath)
        {
            List<(double[,], byte)> data = new List<(double[,], byte)>();
      
            try
            {
                Debug.WriteLine("\nBegin\n");
                FileStream ifsLabels =
                 new FileStream(labelPath,
                 FileMode.Open); // test labels
                FileStream ifsImages =
                 new FileStream(dataPath,
                 FileMode.Open); // test images

                BinaryReader brLabels =
                 new BinaryReader(ifsLabels);
                BinaryReader brImages =
                 new BinaryReader(ifsImages);

                int magic1 = brImages.ReadInt32(); // discard
                int numImages = brImages.ReadInt32();
                int numRows = brImages.ReadInt32();
                int numCols = brImages.ReadInt32();

                int magic2 = brLabels.ReadInt32();
                int numLabels = brLabels.ReadInt32();

                double[,] pixels = new double[28,28];

                // each test image
                for (int di = 0; di < size; ++di)
                {
                    for (int i = 0; i < 28; ++i)
                    {
                        for (int j = 0; j < 28; ++j)
                        {
                            int b = brImages.ReadByte();
                            pixels[i,j] = (b/255.0);
                        }
                    }

                    byte lbl = brLabels.ReadByte();
                    double[,] copy = pixels.Clone() as double[,];
                    data.Add((copy, lbl));
                    Console.ReadLine();
                } // each image

                ifsImages.Close();
                brImages.Close();
                ifsLabels.Close();
                brLabels.Close();

                Debug.WriteLine("\nEnd\n");
                Console.ReadLine();
            }
            catch (Exception ex)
            {
                Debug.WriteLine(ex.Message);
                Console.ReadLine();
            }


            return data;
        }

        public static void serializeData()
        {
            var training_data = readData(60000, "../../../train-labels.idx1-ubyte", "../../../train-images.idx3-ubyte");
            var validation_data = training_data.GetRange(50000, 10000);//readData(60000, "../../../train-labels.idx1-ubyte", "../../../train-images.idx3-ubyte");
            var test_data = readData(10000, "../../../t10k-labels.idx1-ubyte", "../../../t10k-images.idx3-ubyte");
            SerializeNow(training_data.GetRange(0, 50000), "../../../trainingData.nn");
            SerializeNow(validation_data, "../../../validationData.nn");
            SerializeNow(test_data, "../../../testingData.nn");
        }

        public static void SerializeNow(List<(double[,], byte)> data, string fileName)
        {
            FileStream f =
                 new FileStream(fileName,
                 FileMode.Create);
            BinaryFormatter b = new BinaryFormatter();
            b.Serialize(f, data);
            f.Close();
        }
        public static List<(double[,], byte)> DeSerializeNow(string fileName)
        {
            FileStream f =
                 new FileStream(fileName,
                 FileMode.Open);
            BinaryFormatter b = new BinaryFormatter();
            List<(double[,], byte)> c = (List<(double[,], byte)>)b.Deserialize(f);
            Debug.WriteLine(c.Count);
            f.Close();
            return c;
        }

    }
}

