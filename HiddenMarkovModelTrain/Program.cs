using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HiddenMarkovModelTrain
{
    class Program
    {
        static void Main(string[] args)
        {
            int nGestures = 10;
            int nSamples = 55;
            
            logHMM [] samples = new logHMM[nSamples];

            for (int i = 4; i <= 4; i++)
            {
               samples[i] = new logHMM(9, 12);

                for (int j = 1; j <= 3; j++)
                {
                    try
                    {
                        string fileName = "Digit_" + i + "_" + j + ".txt";
                        using (StreamReader sr = new StreamReader(fileName))
                        {

                            String line = sr.ReadToEnd();
                            // Console.WriteLine(line);
                            
                            int [] digit = StringToIntArray(line);

                            samples[i].baumWelchTrain(digit);

                            /* 
                            Console.WriteLine("Digit: \n");
                            Console.WriteLine(digit.Sum().ToString());
                            Console.WriteLine("Digit 0 : " + digit[13].ToString());
                            */
                            

                        }
                    }
                    catch (Exception e)
                    {
                        Console.WriteLine("The file could not be read:");
                        Console.WriteLine(e.Message);
                    }
                }

                Console.WriteLine("----");
            }
                Console.ReadLine();

        }

        private static int[] StringToIntArray(string myNumbers)
        {
            List<int> myIntegers = new List<int>();
            Array.ForEach(myNumbers.Split(",".ToCharArray()), s =>
            {
                int currentInt;
                if (Int32.TryParse(s, out currentInt))
                    myIntegers.Add(currentInt);
            });
            return myIntegers.ToArray();
        }
    }
}
