using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HiddenMarkovModelTrain
{
    class logHMM
    {
        public int numStates; // number of states

        public int numObservations; // size of the observations

        public double[] prior; // initial states (prior)

        public double[,] A; // transitionMatrix

        public double[,] B; // emissionMatrix


        public double[,] fwd; // forward-vars
        public double[,] bwd; // backward-vars

        public double log_prob_forward;

        public int time_duration;

        /** Constructor
         * @ param numStates
         * @ param numObservations
         **/
        public logHMM(int numStates, int numObservations)
        {
            prior = new double[numStates];

            A = new double[numStates, numStates];
            B = new double[numStates, numObservations];
        }


        /** Baum-Welch Training 
         * @param data : the training set
         * @param nSteps: the number of steps
         **/
        public void baumWelchTrain(int[] data, int nSteps)
        {
            time_duration = data.Length;

            double [,] gamma = new double [numStates, time_duration];

            double[,] alpha_num = new double[numStates, numStates]; // new A (transition matrix)
            double [] alpha_denom = new double [numStates];

            double[,] beta_num = new double[numStates, numObservations]; // new B (emission matrix)
            double [,] beta_denom = new double [numStates, numObservations];

            double[] a0_num = new double[numStates]; // new prior 
            double[] a0_denom = new double[numStates]; 



            //calculate the foward and backward variables on the model

            fwd = forwardCalc(data);
            bwd = backwardCalc(data);

            /** RE-ESTIMATION and UPDATE **/

            // re-estimation of the prior=a0 (initial state) probabilities and UPDATE
            for (int i = 0; i < numStates; i++)
            {
                a0_num[i] = fwd[0, i] + bwd[0, i];

                for (int j = 0; j<numStates; j++){
                    double gamma_t_j = fwd[0, i] + bwd [0, i];
                    a0_denom [i] = log_Sum (a0_denom[i], gamma_t_j);
                }

                // Update prior with a0:
                // which means prior = a0;
                prior[i] = a0_num[i] - a0_denom[i];
            }

            // re-estimation of the the transtition prob
            for (int t = 0; t<time_duration -2; t++){

                for (int i=0; i<numStates; i++){
                    alpha_denom[i] = double.NaN;

                    for (int j = 0; j<numStates; j++){
                        alpha_num[i, j] = double.NaN;
                        

                        double xi = fwd[t, i] + A[i, j] + B[j, data[t + 1]] + bwd[t + 1, j] - totalNumberOfTransCalc (t, i);
                        alpha_num [i,j] = log_Sum (alpha_num [i,j], xi);
                        alpha_denom[i] = log_Sum (alpha_denom[i], xi);
                    }
                }
            }

            // update A with alpha: A = alpha

            for (int i = 0; i<numStates; i++){
                
                for (int j = 0; j<numStates; j++){

                    A [i, j] = alpha_num[i,j] - alpha_denom[i]; 
                }
            }

            // re-estimaton of emission prob 

            for (int j = 0; j<numStates; j++){

                for (int k = 0; k<numObservations; k++){
                    beta_num[j, k] = double.NaN;
                    beta_denom[j, k] = double.NaN;

                    for (int t = 0; t < time_duration; t++)
                    {
                        double gamma_t_j = fwd[t, j] + bwd[t, j] - totalNumberOfTransCalc (t, j);

                        // if data[t] is equal to k, then multiply gamma_t_j by 1, otherwise 0

                        if (data[t] == k ){
                            beta_num[j,k] = log_Sum (beta_num[j,k], gamma_t_j);
                        }
                        beta_denom[j, k] = log_Sum(beta_denom[j, k], gamma_t_j);

                    }

                    // UPDATE B [j,k] with beta [j,k]
                    B[j, k] = beta_num[j, k] - beta_denom[j, k]; 
                    
                }
            }

        }


        /** Algorithm for Forward-Variable(s) Calculatiın at the state i, time t
         * @param data -> integer olarak geliyor
         * @return an array, fwd, that contains the forward-variables
         * multi-dimens olduguna dikkat et!!! 
         **/
        public double[,] forwardCalc(int[] data)
        {
            int time_duration = data.Length;

            double[,] fwd = new double[time_duration, numStates];

            for (int t = 0; t < time_duration; t++)
            {
                for (int i = 0; i < numStates; i++)
                {
                    fwd[t, i] = double.NegativeInfinity;
                }
            }

            // at time t = 0
            // initialization of the initial prob states
            for (int i = 0; i < numStates; i++)
            {
                fwd[0, i] = prior[i] + B[i, data[0]];
            }

            // induction step
            for (int t = 0; t < time_duration - 1; t++) // 0 <= t <= T-2
            {
                for (int j = 0; j < numStates; j++)
                {
                    for (int i = 0; i < numStates; i++)
                    {
                        double prob = fwd[t, i] + A[i, j] + B[j, data[t + 1]];
                        fwd[t + 1, j] = log_Sum(fwd[t + 1, j], prob);
                    }
                }
            }

            log_prob_forward = double.NaN;

            for (int i = 0; i < numStates; i++)
            {
                log_prob_forward = log_Sum(log_prob_forward, fwd[time_duration - 1, i]);
            }

            return fwd;
        }

        /** Algorith m for the Backward-Variables(s) Calculation at the state i, time t
         * @ param data -> int alıyor
         * @ return an array, bwd, that contains the backward-variables
         * multi-dimension olduguna dikkat et
         **/
        public double[,] backwardCalc(int[] data)
        {
            int time_duration = data.Length;

            double[,] bwd = new double[time_duration, numStates];

            for (int t = 0; t < time_duration; t++)
            {
                for (int i = 0; i < numStates; i++)
                {
                    bwd[t, i] = double.NegativeInfinity;
                }
            }

            // at time t = 0 -> daha doğrusu t = t (amma velakin beta oldugundan t = 0)
            // initialization 
            for (int i = 0; i < numStates; i++)
            {
                bwd[time_duration - 1, i] = 0; // log (1) = 0; -> normalde 1 olacakti ancak simdi 0 olacak log'dan dolayi
            }

            // induction step
            for (int t = time_duration - 2; t >= 0; t--)
            {
                for (int i = 0; i < numStates; i++)
                {
                    for (int j = 0; j < numStates; j++)
                    {
                        double prob = A[i, j] + B[j, data[t + 1]] + bwd[t + 1, j];
                        bwd[t, i] += log_Sum(bwd[t,i], prob);
                    }
                }
            }

            return bwd;
        }

        double log_Sum (double p1, double p2)
        {
            if (p1 > p2)
            {
                return p1 + Math.Log(Math.Exp(p2 - p1));
            }
            else
            {
                return p2 + Math.Log(Math.Exp(p1 - p2));
            }
        }

        double totalNumberOfTransCalc (int t, int i)
        {
            double total_trans_num = fwd[t, i] + bwd[t, i];
            double total_trans_denom = double.NaN;

            for (int j = 0; j < numStates; j++)
            {
                total_trans_denom = log_Sum(total_trans_denom, (fwd[t, j] + bwd[t, j]));
            }

            return total_trans_num - total_trans_denom;
        }

    }
}
