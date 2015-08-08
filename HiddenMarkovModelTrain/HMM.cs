using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HiddenMarkovModelTrain
{
    class HMM
    {
        public int numStates; // number of states

        public int numObservations; // size of the observations

        public double [] prior; // initial states (prior)

        public double [,] A; // transitionMatrix

        public double [,] B; // emissionMatrix

        /** Constructor
         * @ param numStates
         * @ param numObservations
         **/
        public HMM(int numStates, int numObservations)
        {
            prior = new double[numStates];

            A = new double[numStates, numStates];
            B = new double[numStates, numObservations];
        }


        /** Baum-Welch Training 
         * @param data : the training set
         * @param nSteps: the number of steps
         **/
        public void baumWelchTrain (int [] data, int nSteps)
        {
            int time_duration = data.Length;

            double [,] alpha = new double [numStates, numStates];
            double[,] beta = new double[numStates, numObservations];
            double[] a0 = new double[numStates];

            double[,] fwd;
            double[,] bwd;

            for (int step = 0; step<nSteps; step++){

                //calculate the foward and backward variables on the model

                fwd = forwardCalc (data);
                bwd = backwardCalc (data);

                // RE-ESTIMATION

                // re-estimation of the prior=a0 (initial state) probabilities
                for (int i = 0; i<numStates; i++){
                    a0 [i] = gamma (i, 0, data, fwd, bwd);
                }

                // re-estimation of the the transtition prob
                for (int i = 0; i<numStates; i++){
                    for (int j = 0; j<numStates; j++){
                        double numerator = 0;
                        double denominator = 0;

                        for (int t = 0; t<time_duration; t++){
                            numerator += probCalc (t, i, j, data, fwd, bwd);
                            denominator += gamma (i, t, data, fwd, bwd);
                        }
                        alpha [i, j] = divide(numerator, denominator);
                    }
                }


                // re-estimaton of emission prob

                for (int i = 0; i<numStates; i++){
                    for (int k = 0; k<numObservations; k++){

                        double numerator = 0;
                        double denominator = 0;

                        for (int t = 0; t<time_duration; t++){
                            double e = gamma (i, t, data, fwd, bwd);

                            // if data[t] is equal to k, then multiply e by 1, otherwise 0
                            numerator += e * (data[t] == k ? 1 : 0); 
                            denominator += e;
                        }

                        beta [i, k] = divide(numerator, denominator);
                    }
                }

                prior = a0;
                A = alpha;
                B = beta;
            }
        }


        /** Algorithm for Forward-Variable(s) Calculatiın at the state i, time t
         * @param data -> integer olarak geliyor
         * @return an array, fwd, that contains the forward-variables
         * multi-dimens olduguna dikkat et!!! 
         **/
        public double [, ] forwardCalc (int [] data)
        {
            int time_duration = data.Length;

            double[,] fwd = new double[numStates, time_duration];

            // at time t = 0
            // initialization of the initial prob states
            for (int i = 0; i < numStates; i++)
            {
                fwd[i, 0] = prior[i] * B[i, data[0]];
            }

            // induction step
            for (int t = 0; t<time_duration - 1; t++) // 0 <= t <= T-2
            {
                for (int j = 0; j < numStates; j++)
                {
                    fwd[j, t + 1] = 0.0; // ne olur ne olmaz diye 0 degil de 0.0 yaptım

                    for (int i = 0; i < numStates; i++)
                    {
                        fwd[j, t + 1] += (fwd[i, t] * A[i, t]);
                    }

                    fwd[j, t + 1] *= B[j, data[t + 1]];
                }
            }

            return fwd;
        }

        /** Algorith m for the Backward-Variables(s) Calculation at the state i, time t
         * @ param data -> int alıyor
         * @ return an array, bwd, that contains the backward-variables
         * multi-dimension olduguna dikkat et
         **/
        public double [,] backwardCalc (int [] data)
        {
            int time_duration = data.Length;

            double[,] bwd = new double[numStates, time_duration];

            // at time t = 0 -> daha doğrusu t = t (amma velakin beta oldugundan t = 0)
            // initialization 
            for (int i = 0; i < numStates; i++)
            {
                bwd[i, time_duration - 1] = 1; // hepsi 1
            }

            // induction step
            for (int t = time_duration - 2; t >= 0; t--)
            {
                for (int i = 0; i < numStates; i++)
                {
                    bwd[i, t] = 0.0; // ne olur ne olmaz diye 0 degil de 0.0 yaptım

                    for (int j = 0; j < numStates; j++)
                    {
                        bwd[i, t] += (bwd[j, t + 1] * A[i, j] * B[j, data[t + 1]]);
                    }
                }
            }

            return bwd;
        }

        /** Calculates Gamma
         * @ param"s": i ,t data, fwd, bwd
         * @return gamma
         **/
        public double gamma (int i, int t, int [] data, double [,] fwd, double [,] bwd)
        {
            double numerator = fwd[i, t] * bwd[i, t];
            double denominator = 0.0; // ne olur ne olmaz diye... = 0.0

            for (int j = 0; j < numStates; j++)
            {
                denominator += (fwd[j, t] * bwd[j, t]);
            }

            return divide(numerator, denominator);
        }


        /** Calculates the probability p
         * @ param t: time
         * @ param i: s_i
         * @ param j: s_j
         * @ param data: data (Obs sequence)
         * @ param fwd: forward
         * @ param bwd: backward
         * @ return probabiliy p
         **/
        double probCalc (int t, int i, int j, int [] data, double [, ] fwd, double [,] bwd)
        {
            double numerator;
            double denominator = 0;

            int time_duration = data.Length;

            // if t == T -1 -> num = fwd(i,t) * A[i,j])
            // else num = fwd(i,t) * A[i,j] * B [j, data[t+1] * bwd[j, t+1]
            numerator = (t == time_duration -1 ? (fwd[i,t] * A[i,j]) : (fwd[i,t] * A[i,j] * B [j, data[t+1]] * bwd[j, t+1]));

            //denom calc
            for (int k = 0; k<numStates; k++){
                denominator += (fwd[k,t] * bwd[k,t]);
            }

            return divide(numerator, denominator);
        }


        /**Divides two double numbers
         * @ param numerator
         * @ oaram denominatr
         * @return num / denum -> float
         **/
        double divide (double num, double denum)
        {
            if (num == 0)
            {
                return 0;
            }
            else
            {
                return num / denum;
            }
        }
    }
}
