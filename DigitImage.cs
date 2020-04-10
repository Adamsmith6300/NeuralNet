using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Numpy;

namespace NeuralNet2
{
    class DigitImage
    {
        public byte[][] pixelsBytes;
        public byte labelByte;
        public NDarray pixels;
        public NDarray label;

        public DigitImage(byte[][] pixels,
          byte label)
        {
            this.pixelsBytes = new byte[28][];
            for (int i = 0; i < this.pixelsBytes.Length; ++i)
                this.pixelsBytes[i] = new byte[28];

            for (int i = 0; i < 28; ++i)
                for (int j = 0; j < 28; ++j)
                    this.pixelsBytes[i][j] = pixels[i][j];

            //this.pixels = np.array(this.pixelsBytes);
            this.labelByte = label;
            var a = new int[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
            a[label] = 1;
            //this.label = np.array(a);
        }

        public override string ToString()
        {
            string s = "";
            for (int i = 0; i < 28; ++i)
            {
                for (int j = 0; j < 28; ++j)
                {
                    if (this.pixelsBytes[i][j] == 0)
                        s += " "; // white
                    else if (this.pixelsBytes[i][j] == 255)
                        s += "O"; // black
                    else
                        s += "."; // gray
                }
                s += "\n";
            }
            s += this.label.ToString();
            return s;
        }
    }
}
